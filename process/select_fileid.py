import os 
import numpy as np
from multiprocessing import Pool
import multiprocessing as mp
import argparse
import time
import torch.distributed as dist
import math
import torch
import sys 
import pdb 
sys.path.append('.')
sys.path.append('..')
from utils.utils import compute_cosine,compute_graph,add_list

def work(clip_names,save_path):
    i = 0
    unique_video_dict = {}
    for clip_name in clip_names:
        keys = [os.path.join(clip_name,f) for f in os.listdir(clip_name)]
        if len(keys) < 5:
            continue
        feature_matrix = np.array([np.load(k) for k in keys])[:,0] 
        

        # compute face id cos
        cos_dis = compute_cosine(feature_matrix)

        # build graph
        repeat_graph,index = compute_graph(cos_dis)

        repeat_keys = add_list([v for v in repeat_graph.values()])

        if repeat_keys is None:
            repeat_keys = []

        assert(len(set(index[0].tolist()+index[1].tolist())) == len(repeat_keys))

        # store the video in dict by same id
        
        for v in repeat_graph.values():
            vals = np.array(keys)[v]
            
            unique_video_dict[i] = vals.tolist()
            i += 1
            print('\r have done %06d'%i,end='',flush=True)
        for v in [v for v in range(len(feature_matrix)) if v not in repeat_keys]:
            vals = [keys[v]]
            
            unique_video_dict[i] = vals
            
            i += 1
            print('\r have done %06d'%i,end='',flush=True)   
    print()
    np.save(save_path,unique_video_dict)

def print_error(value):
    print("error: ", value)

parser = argparse.ArgumentParser(description="HeadSwap")
parser.add_argument('--pool_num',default=10,type=int)

if __name__ == "__main__":
    args = parser.parse_args()
    mp.set_start_method('spawn')

    base = '/mnt/user/zhaoxiang/workspace/FaceSwap/DATA/HeadSwap/wav2lip-headswap/id/'
    clip_names = [os.path.join(base,f) for f in os.listdir(base)]
    save_base = '/mnt/user/zhaoxiang/workspace/FaceSwap/DATA/HeadSwap/wav2lip-headswap/info'
    os.makedirs(save_base,exist_ok=True)

    rank = int(os.environ.get('RANK','0'))
    world_size = int(os.environ.get('WORLD_SIZE','1'))
    print('*********************',rank,world_size)

    pool_num = args.pool_num
    length = len(clip_names)
    dis1 = math.ceil(length / float(world_size))
    clip_names = clip_names[rank*dis1:(rank+1)*dis1]


    length = len(clip_names)
    dis = math.ceil(length/float(pool_num))
    
    if world_size > 1:
        dist.init_process_group(backend="nccl") # backbend='nccl'
        dist.barrier() # 用于同步训练
    signal = torch.tensor([0]).cuda()
    
    t1 = time.time()
    # i = 0
    # save_path = os.path.join(save_base,str(rank)+'-'+str(i)+'.npy')
    # work(clip_names,save_path)
    print('***************all length: %d ******************'%length)
    p = Pool(pool_num)
    for i in range(pool_num):
        save_path = os.path.join(save_base,str(rank)+'-'+str(i)+'.npy')
        p.apply_async(work, args = (clip_names[i*dis:(i+1)*dis],save_path,),error_callback=print_error)   
    p.close() 
    p.join()
    print("all the time: %s"%(time.time()-t1))

    signal = torch.tensor([1]).cuda()
    if world_size > 1:
        while True:

            dist.all_reduce(signal)
            value = signal.item()
            print('***************',value)
            if value >= world_size:
                break 
            else:
                dist.all_reduce(torch.tensor([0]).cuda())
    
    
    if rank == 0:
        id_dict = {}
        i = 0
        file_paths = [os.path.join(save_base,f) for f in os.listdir(save_base)]
        for f in file_paths:
            for k,v in np.load(f,allow_pickle=True).item().items():
                id_dict[i] = v 
                i += 1
        np.save(os.path.join(save_base,'single_id.npy'),id_dict)