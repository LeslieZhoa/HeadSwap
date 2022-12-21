import sys 
import os
import numpy as np
import cv2
import pdb
from multiprocessing import Pool
import multiprocessing as mp
import argparse
import time
import torch.distributed as dist
import math
import torch
sys.path.append('..')
sys.path.append('.')
from model.third.Deep3dRec.get_params import FaceRec

def work(img_bases):
    face_rec = FaceRec('../pretrained_models/epoch_20.pth',
                    '../pretrained_models/BFM',use_lmk=True)
    k = 1
    for img_base in img_bases:
        save_path = img_base.replace('img','crop')
        bbox_path = os.path.join(img_base,'box.npy')
        if not os.path.exists(bbox_path):
            continue

        os.makedirs(save_path,exist_ok=True)
        os.makedirs(save_path.replace('crop','3dmm'),exist_ok=True)

        bbox = np.load(bbox_path)
        x_min,y_min,x_max,y_max = bbox[:4]
        
        center_x = (x_min + x_max) / 2.0
        center_y = (y_min + y_max) / 2.0
        bbox_size = int(max(y_max-y_min,x_max-x_min) * 1.8)

        x_min = int(center_x-bbox_size * 0.5)
        y_min = int(center_y-bbox_size * 0.5)
        x_max = x_min + bbox_size
        y_max = y_min + bbox_size
        boundingBox = None
        
        for img_name in [f for f in os.listdir(img_base) if f.endswith('.png')]:
            if os.path.exists(os.path.join(save_path,img_name)):
                continue
            
            img = cv2.imread(os.path.join(img_base,img_name))
            if boundingBox is None:
                h,w,_ = img.shape 
                boundingBox = [max(x_min, 0), max(y_min, 0), min(x_max, w), min(y_max, h)]
            imgCropped = img[boundingBox[1]:boundingBox[3], boundingBox[0]:boundingBox[2]]
            imgCropped = cv2.copyMakeBorder(imgCropped, max(-y_min, 0), max(y_max - h, 0), max(-x_min, 0),
                                        max(x_max - w, 0),cv2.BORDER_CONSTANT,value=(0,0,0))

            params = face_rec.run(cv2.cvtColor(
                            cv2.resize(imgCropped,(256,256)),
                            cv2.COLOR_BGR2RGB))
            cv2.imwrite(os.path.join(save_path,img_name),cv2.resize(imgCropped,(512,512)))
            np.save(os.path.join(save_path.replace('crop','3dmm'),img_name.replace('.png','.npy')),params)
            print('\r have done %06d'%k,end='',flush=True)
            k += 1

parser = argparse.ArgumentParser(description="HeadSwap")

parser.add_argument('--pool_num',default=10,type=int)

if __name__ == "__main__":
    args = parser.parse_args()
    mp.set_start_method('spawn')
    
    base = 'HeadSwap/wav2lip-headswap/img'
    img_bases = [os.path.join(base,f) for f in os.listdir(base)]

    rank = int(os.environ.get('RANK','0'))
    world_size = int(os.environ.get('WORLD_SIZE','1'))
    print('*********************',rank,world_size)

    pool_num = args.pool_num
    length = len(img_bases)
    dis1 = math.ceil(length / float(world_size))
    img_bases = img_bases[rank*dis1:(rank+1)*dis1]


    length = len(img_bases)
    dis = math.ceil(length/float(pool_num))
    
    if world_size > 1:
        dist.init_process_group(backend="nccl") # backbend='nccl'
        dist.barrier() # 用于同步训练
    signal = torch.tensor([0]).cuda()
    
    t1 = time.time()
    print('***************all length: %d ******************'%length)
    p = Pool(pool_num)
    for i in range(pool_num):
        p.apply_async(work, args = (img_bases[i*dis:(i+1)*dis],))   
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
                signal = torch.tensor([1]).cuda()
    