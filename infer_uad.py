import os
import json
import argparse
import numpy as np
import h5py
import torch 
import soundfile as sf
from tqdm import tqdm
from scipy.special import softmax
from python_speech_features import logfbank
from module.model import Gvector

# config:
mdl_clf_kwargs = {
    "channels": 16, 
    "block": "BasicBlock", 
    "num_blocks": [2,2,2,2], 
    "embd_dim": 1024, 
    "drop": 0.3, 
    "n_class": 5
}

# config:
mdl_uad_kwargs = {
    "channels": 16, 
    "block": "BasicBlock", 
    "num_blocks": [2,2,2,2], 
    "embd_dim": 1024, 
    "drop": 0.3, 
    "n_class": 2
}

def parse_args():
    desc="infer labels"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--dataset', type=str, default=None, help="path to the dataset dir for inference")
    parser.add_argument('--output', type=str, default=None, help="path to the output dir")
    parser.add_argument('--model-clf', type=str, required=True)
    parser.add_argument('--model-uad', type=str, required=True)
    return parser.parse_args()
    

class SVExtractor():
    def __init__(self, mdl_kwargs, model_path, device):
        self.model = self.load_model(mdl_kwargs, model_path)
        self.model.eval()
        self.device = device
        self.model = self.model.to(self.device)

    def load_model(self, mdl_kwargs, model_path):
        model = Gvector(**mdl_kwargs)
        state_dict = torch.load(model_path)
        if 'model' in state_dict.keys():
            state_dict = state_dict['model']
        model.load_state_dict(state_dict)
        return model

    def extract_logfbank(self, feat_path, cmn=True):
        hf = h5py.File(feat_path, 'r')
        logfbankFeat = np.array(hf.get('logfbank'))
        hf.close()
        if cmn:
            logfbankFeat -= logfbankFeat.mean(axis=0, keepdims=True)
        return logfbankFeat.astype('float32')

    def __call__(self, feat_path):
        feat = self.extract_logfbank(feat_path)
        feat = torch.from_numpy(feat).unsqueeze(0)
        feat = feat.float().to(self.device)
        with torch.no_grad():
            embd = self.model(feat)
        embd = embd.squeeze(0).cpu().numpy()
        return embd


if __name__ == "__main__":
    args = parse_args()
    model_clf_path = args.model_clf
    model_uad_path = args.model_uad
    dataset_dir = args.dataset.rstrip('/') + '/'
    output_dir = args.output.rstrip('/') + '/'
    os.makedirs(output_dir, exist_ok=True)
    print('... loading model ...')
    clf_extractor = SVExtractor(mdl_clf_kwargs, model_clf_path, device='cpu')
    uad_extractor = SVExtractor(mdl_uad_kwargs, model_uad_path, device='cpu')
    print('... loaded ...')
    all_wavs = [dataset_dir+wav for wav in os.listdir(dataset_dir) if wav.endswith('.h5')]
    predictions = []
    for wav in tqdm(all_wavs, desc=dataset_dir.split('/')[-2]):
        isUnseen = np.argmax(softmax(uad_extractor(wav)))
        if isUnseen:
            predictions.append("%s, %d"%(wav.split('/')[-1].replace('.h5','.wav'), 5))
        else:
            embd = softmax(clf_extractor(wav))
            predictions.append("%s, %d"%(wav.split('/')[-1].replace('.h5','.wav'), np.argmax(embd)))
    with open(output_dir+'answer_uad.txt','w') as f:
        f.write('\n'.join(predictions))