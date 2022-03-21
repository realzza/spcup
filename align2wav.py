import sox
import os
import argparse
import numpy as np
from tqdm import tqdm
from multiprocessing import Process

def parse_args():
    desc="align smaplerate of dataset"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--dataDir', type=str, default=None, help="path to the input dir")
    parser.add_argument('--outDir', type=str, default=None, help="path to the output dir")
    parser.add_argument('--process', type=int, default=20, help="number of process running")
    return parser.parse_args()

def align(pid, nproc, allClips, tfm, bad_path):
    # set flag to record killing session
    isFirst = True
    
    # assign portion to each process
    portion = len(allClips) // nproc
    if pid == 0:
        allClips = allClips[:portion]
    elif pid == nproc - 1:
        allClips = allClips[portion*pid:]
    else:
        allClips = allClips[portion*pid: portion*(pid+1)]

    for song in tqdm(allClips, desc='process %d'%os.getpid()):
        if isFirst:
            with open('kill_align.sh','a') as f:
                f.write('kill -9 %d\n'%os.getpid())
            isFirst = False
            
        name = song.split('/')[-1]
        try:
            tfm.build_file(song, outDir + name[:-4] + '.wav')
        except:
            # record bad transforms on the fly
            with open(bad_path,'a') as f:
                f.write(song + '\t' + outDir + name[:-4] + '.wav\n')
    
    
if __name__ == '__main__':
    
    # define a transformer
    tfm = sox.Transformer()
    tfm.set_output_format(file_type='wav', rate=16000, bits=16, channels=1)
    
    # load args and clips
    args = parse_args()
    dataDir = args.dataDir.rstrip('/') + '/'
    allClips = [dataDir + x for x in os.listdir(dataDir) if x.endswith('mp3')]
        
    # init output directories
    outDir = args.outDir.rstrip('/') + '/'
    os.makedirs(outDir, exist_ok=True)
        
    # initialize kill script and bad file records
    with open('kill_align.sh','w') as f:
        f.write('')
        
    bad_file_path = 'bad_aligns.txt'
    with open(bad_file_path,'w') as f:
        f.write('')

    # multiprocessing transformation
    worker_count = args.process
    worker_pool = []
    for i in range(worker_count):
        p = Process(target=align, args=(i, worker_count, allClips, tfm, bad_file_path))
        p.start()
        worker_pool.append(p)
    for p in worker_pool:
        p.join()  # Wait for all of the workers to finish.

    # Allow time to view results before program terminates.
    os.remove('kill_align.sh')
    a = input("Finished")