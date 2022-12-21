import os 
import imageio
import numpy as np
from face_alignment.detection.sfd import FaceDetector
import torch
import math
import cv2
import pdb
from multiprocessing import Pool
import multiprocessing as mp
import argparse
import sys 
import time
import torch.distributed as dist
sys.path.insert(0,'.')
sys.path.insert(0,'..')
sys.path.insert(0,'BaseModel/LVT/')
from LVT import Engine
import face_utils
import pdb            

def process_frame(video_paths,save_base,thresh):
    face_detector = FaceDetector(device='cuda')
    engine = Engine(
                    face_id_path='LVT-Model/id_model.onnx')
    def detect_faces(images):
        images = np.stack(images).transpose(0,3,1,2).astype(np.float32)
        images_torch = torch.tensor(images)
        return face_detector.detect_from_batch(images_torch.cuda())

    for video_path in video_paths:
        video_name = video_path.split('/')[-2] + '-' + os.path.splitext(os.path.basename(video_path))[0]
        video_reader = imageio.get_reader(video_path)
        index = 0
        kk = 0
        save_path = os.path.join(save_base,video_name+'-'+str(index))
        os.makedirs(save_path,exist_ok=True)
        os.makedirs(save_path.replace('img','bbox'),exist_ok=True)
        os.makedirs(save_path.replace('img','id'),exist_ok=True)
        tube_bbox = None
        for img in video_reader:
            
            bboxes = detect_faces([img])[0]
            
            bboxes = list(filter(lambda x:x[-1]>0.99,bboxes))

            if len(bboxes) == 0:
                continue 
            
            bbox = find_bigest_box(bboxes)

            iou_score = compute_iou(tube_bbox,bbox)
            if iou_score < thresh:
                np.save(os.path.join(save_path,'box.npy'),tube_bbox)
                index += 1
                kk = 0
                save_path = os.path.join(save_base,video_name+'-'+str(index))
                os.makedirs(save_path,exist_ok=True)
                os.makedirs(save_path.replace('img','bbox'),exist_ok=True)
                os.makedirs(save_path.replace('img','id'),exist_ok=True)
                tube_bbox = None

            if tube_bbox is None:
                tube_bbox = bbox 
            else:
                tube_bbox = merge_box(tube_bbox,bbox)
              
            # get id feature
            img_crop = img[int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2])]
            id_inp = face_utils.utils.preprocess(img_crop[...,::-1],size=[112,112],mean=None,std=None)
            id_feature = engine.get_id(id_inp)

            np.save(os.path.join(save_path.replace('img','id'),'%s-%04d.npy'%(video_name,kk)),id_feature)
            cv2.imwrite(os.path.join(save_path,'%s-%04d.png'%(video_name,kk)),img[...,::-1])
            # cv2.imwrite(os.path.join(save_path,'%s-%04d.png'%(video_name,kk)),cv2.rectangle(img,
            #         (int(bbox[0]),int(bbox[1])),(int(bbox[2]),int(bbox[3])),[0,0,255],3)[...,::-1])
            np.save(os.path.join(save_path.replace('img','bbox'),'%s-%04d.npy'%(video_name,kk)),bbox)
            kk += 1      
            print('\r have done %06d'%kk,end='',flush=True)
        np.save(os.path.join(save_path,'box.npy'),tube_bbox)  
        video_reader.close()
    print()

def merge_box(tube_bbox, bbox):
    xA = min(tube_bbox[0], bbox[0])
    yA = min(tube_bbox[1], bbox[1])
    xB = max(tube_bbox[2], bbox[2])
    yB = max(tube_bbox[3], bbox[3])
    return (xA, yA, xB, yB)


def find_bigest_box(bboxes):
    if len(bboxes) == 1:
        return bboxes[0]
    bboxes = np.array(bboxes)
    areas = (bboxes[:,2] - bboxes[:,0]) * (bboxes[:,3] - bboxes[:,1])
    return bboxes[np.argmax(areas)]
    
def compute_iou(boxA,boxB):
    if boxA is None or boxB is None:
        return 1
    boxA = [int(x) for x in boxA]
    boxB = [int(x) for x in boxB]

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    
    iou = interArea / float(boxAArea + boxBArea - interArea)

    return iou

parser = argparse.ArgumentParser(description="HeadSwap")

parser.add_argument('--pool_num',default=10,type=int)

if __name__ == "__main__":
    args = parser.parse_args()
    mp.set_start_method('spawn')
    thresh = 0.5
    base = 'head_swap'

    clip_paths = [os.path.join(base,f) for f in os.listdir(base)]
    video_paths = [os.path.join(clip_path,f) for clip_path in clip_paths for f in os.listdir(clip_path)]

    save_base = 'HeadSwap/wav2lip-headswap/img'
    # save_base = '../../dataset/wav2lip-test'
    
    
    rank = int(os.environ.get('RANK','0'))
    world_size = int(os.environ.get('WORLD_SIZE','1'))
    print('*********************',rank,world_size)
  
   
    pool_num = args.pool_num
    length = len(video_paths)
    dis1 = math.ceil(length / float(world_size))
    video_paths = video_paths[rank*dis1:(rank+1)*dis1]


    length = len(video_paths)
    dis = math.ceil(length/float(pool_num))
    
    if world_size > 1:
        dist.init_process_group(backend="nccl") # backbend='nccl'
        dist.barrier() # 用于同步训练
    signal = torch.tensor([0]).cuda()
    
    t1 = time.time()
    print('***************all length: %d ******************'%length)
    p = Pool(pool_num)
    for i in range(pool_num):
        p.apply_async(process_frame, args = (video_paths[i*dis:(i+1)*dis],save_base,thresh,))   
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
    
