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
from module.model import Gvector, Gvector_rnn

# config:
mdl_kwargs = {
    "channels": 16, 
    "block": "BasicBlock", 
    "num_blocks": [2,2,2,2], 
    "embd_dim": 1024, 
    "drop": 0.3, 
    "n_class": 6
}

mdl_rnn_kwargs = {
    "channels": 16, 
    "block": "BasicBlock", 
    "num_blocks": [2,2,2,2], 
    "embd_dim": 1024, 
    "rnn_dim": 80,
    "rnn_layers": 1,
    "drop": 0.3, 
    "n_class": 6
}

fbank_kwargs = {
    "winlen": 0.025, 
    "winstep": 0.01, 
    "nfilt": 80, 
    "nfft": 2048, 
    "lowfreq": 50, 
    "highfreq": None, 
    "preemph": 0.97    
}

def parse_args():
    desc="infer labels"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--dataset', type=str, default=None, help="path to the dataset dir for inference")
    parser.add_argument('--output', type=str, default=None, help="path to the output file")
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--device', type=str, default="cuda:0", help="device to infer")
    parser.add_argument('--rnn', action="store_true", default="store_false")
    return parser.parse_args()


class SVExtractor():
    def __init__(self, mdl_kwargs, fbank_kwargs, model_path, isRnn, device):
        self.model = self.load_model(mdl_kwargs, model_path, isRnn, device)
        self.model.eval()
        self.device = device
        self.model = self.model.to(self.device)
        self.fbank_kwargs = fbank_kwargs

    def load_model(self, mdl_kwargs, model_path, isRnn, device):
        if isRnn:
            model = Gvector_rnn(**mdl_rnn_kwargs)
        else:
            model = Gvector(**mdl_kwargs)
        state_dict = torch.load(model_path, map_location=device)
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
    model_path = args.model
    dataset_dir = args.dataset.rstrip('/') + '/'
    output_dir = '/'.join(args.output.split('/')[:-1])
    os.makedirs(output_dir, exist_ok=True)
    print('... loading model ...')
    sv_extractor = SVExtractor(mdl_kwargs, fbank_kwargs, model_path, args.rnn, device=args.device)
    print('... loaded ...')
    all_wavs = [dataset_dir+wav for wav in os.listdir(dataset_dir) if wav.endswith('.h5')]
    predictions = []
    for wav in tqdm(all_wavs, desc=dataset_dir.split('/')[-2]):
        embd = softmax(sv_extractor(wav))
        predictions.append("%s, %d"%(wav.split('/')[-1].replace('.h5','.wav'),np.argmax(embd)))
    with open(args.output,'w') as f:
        f.write('\n'.join(predictions))