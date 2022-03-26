import os
import audiofile as af
import numpy as np
import h5py
import argparse
from tqdm import tqdm
from python_speech_features import logfbank, fbank, mfcc

# parse args
def parse_args():
    desc="extract features to h5 struct"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--data', type=str, default=None, help="path to the input/output dir")
    parser.add_argument('--output', type=str, default=None, help="path to the output dir")
    parser.add_argument('--nproc', type=int, default=4, help="number of process to extract features")
    return parser.parse_args()

def featExtractWriter(wavPath, cmn=True):
    kwargs = {
        "winlen": 0.025,
        "winstep": 0.01,
        "nfilt": 80,
        "nfft": 2048,
        "lowfreq": 50,
        "highfreq": 8000,
        "preemph": 0.97
    }
    y, sr = af.read(wavPath)
    featMfcc = mfcc(y, sr, winfunc=np.hamming, **kwargs)
    featLogfbank = logfbank(y, sr, **kwargs)
    if cmn:
        featMfcc -= np.mean(featMfcc, axis=0, keepdims=True)
        featLogfbank -= np.mean(featLogfbank, axis=0, keepdims=True)
    return (featMfcc,featLogfbank)

def extract(i, args):
    with open("kill.sh",'a') as f:
        f.write("kill -9 %d\n"%os.getpid())
    
    dataset_dir = args.data.rstrip("/")+'/'
    feats_dir = args.output.rstrip("/")+'/'
    os.makedirs(feats_dir, exist_ok=True)
            
    # init lists for generation
    # tags = [tag for tag in os.listdir(dataset_dir) if not tag.startswith('.')]
    
    # read from index file
    all_wavs = []
    all_wavs += [dataset_dir + wav for wav in os.listdir(dataset_dir) if wav.endswith(".wav")]
    # for tag in tags:
    #     all_wavs += [dataset_dir+ tag+'/' + wav for wav in os.listdir(dataset_dir+tag) if wav.endswith(".wav")]
    #     os.makedirs(feats_dir+tag, exist_ok=True)
    
    # assign tasks
    one_portion = len(all_wavs) // args.nproc
    if i == args.nproc-1:
        all_wavs = all_wavs[i*one_portion:]
    else:
        all_wavs = all_wavs[i*one_portion:(i+1)*one_portion]

    extracted = os.listdir(feats_dir)
    # for tag in tags:
    #     extracted += os.listdir(feats_dir+tag)

    for seg in tqdm(all_wavs, desc="extracting %s"%os.getpid()):
        seg_name = seg.split("/")[-1]
        # tag_name = seg.split("/")[-2]
        if seg_name.replace("wav","h5") in extracted:
            continue
        h5_out = "%s/%s.h5"%(feats_dir, seg_name.split(".")[0])
        try:
            feat_mfcc, feat_logfbank = featExtractWriter(seg)
        except:
            with open('bad_audio','a') as f:
                f.write('%s\n'%seg)
            continue

        tmp_h5 = h5py.File(h5_out,'w')
        tmp_h5.create_dataset('mfcc', data=feat_mfcc)
        tmp_h5.create_dataset('logfbank', data=feat_logfbank)

            
if __name__ == "__main__":
    args = parse_args()
    with open("kill.sh",'w') as f:
        f.write("")
        
    with open('bad_audio', 'w') as f:
        f.write("")
    
    from multiprocessing import Process
    worker_count = args.nproc
    worker_pool = []
    for i in range(worker_count):
        p = Process(target=extract, args=(i,args))
        p.start()
        worker_pool.append(p)
    for p in worker_pool:
        p.join()  # Wait for all of the workers to finish.

    # Allow time to view results before program terminates.
    a = input("Finished")
