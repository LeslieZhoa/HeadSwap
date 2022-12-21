import sys 
import numpy as np
import os 
import cv2
from multiprocessing import Pool
import math
import time
import torch
sys.path.insert(0,'BaseModel/LVT')
from LVT import Engine


class Infer:
    def __init__(self):
        
        self.model = Engine(
                    face_parsing_path='LVT-Model/face_parsing.pt')

        self.pad = np.array([255,255,255])

    def run(self,video_paths):
        i = 0
        for video_path in video_paths:
            save_base = video_path.replace('crop','mask')
            os.makedirs(save_base,exist_ok=True)
            for img_name in os.listdir(video_path):
                img_path = os.path.join(video_path,img_name)
        
                img = cv2.imread(img_path)
                mask = self.run_single(img)
                cv2.imwrite(img_path.replace('crop','mask'),mask)
                print('\rhave done %06d'%i,end='',flush=True)
                i += 1
        print()
    def run_single(self,img):
        # mask = np.zeros_like(img)
        x = self.model.preprocess_parsing(img)
        out = self.model.postprocess_parsing(self.model.get_parsing(x),*img.shape[:2])
        
        return out

def work(video_paths):
    model = Infer()
    model.run(video_paths)

if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')

    base = 'HeadSwap/wav2lip-headswap/crop/'
    video_paths = [os.path.join(base,f) for f in os.listdir(base)]

    i = 0
    
    pool_num = 5
    length = len(video_paths)
   
    dis = math.ceil(length/float(pool_num))
    # work(video_paths[i*dis:(i+1)*dis])
   
    t1 = time.time()
    print('***************all length: %d ******************'%length)
    p = Pool(pool_num)
    for i in range(pool_num):
        p.apply_async(work, args = (video_paths[i*dis:(i+1)*dis],))

    p.close() 
    p.join()
    print("all the time: %s"%(time.time()-t1))
