#! /usr/bin/python 
# -*- encoding: utf-8 -*-
'''
@author LeslieZhao
@date 20221221
'''

import os 
from torchvision import transforms 
import PIL.Image as Image
from dataloader.DataLoader import DatasetBase
import random
import math
import torch
import numpy as np


class AlignData(DatasetBase):
    def __init__(self, slice_id=0, slice_count=1,dist=False, **kwargs):
        super().__init__(slice_id, slice_count,dist, **kwargs)


        self.transform = transforms.Compose([
            transforms.Resize((kwargs['size'], kwargs['size'])),
            transforms.ToTensor(),
            transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
        ])

        self.resize = transforms.Compose([
            transforms.Resize((256,256))])

        # source root
        root = kwargs['root']
        self.idinfo = np.load(root,allow_pickle=True).item()
        keys = list(self.idinfo.keys())
        
        dis = math.floor(len(keys)/self.count)
        self.keys = keys[self.id*dis:(self.id+1)*dis]
        self.length = len(self.keys)
        random.shuffle(self.keys)
        self.eval = kwargs['eval']
        self.size = kwargs['size']
        self.params_w0 = self.params_h0 = 256
        self.params_target_size = 224
       

    def __getitem__(self,i):
        
        src_img_path,\
            tgt_img_path,\
            src_param_path,\
            tgt_param_path,\
            src_box_path,\
            tgt_box_path = self.get_path(i)

        tube_box_path = os.path.join(os.path.split(src_img_path)[0].replace('crop','img'),'box.npy')
        tube_box = np.load(tube_box_path)
        with Image.open(src_img_path) as img:
            xs = self.transform(img.convert('RGB'))
        xs_params = torch.from_numpy(np.load(src_param_path).astype(np.float32))
        xs_bbox =  np.load(src_box_path)
        xs_bbox = torch.from_numpy(
            np.concatenate([self.fix_bbox(xs_bbox,tube_box),
                self.get_params_box(xs_params.numpy())],-1).astype(np.float32))

        flag = 1
        # ÷if self.eval
        if random.random() > 0.5:
            tgt_img_path,tgt_param_path,tgt_box_path = self.get_another_tgt(i)
            tube_box_path = os.path.join(os.path.split(tgt_img_path)[0].replace('crop','img'),'box.npy')
            tube_box = np.load(tube_box_path)
            flag = 0


        with Image.open(tgt_img_path) as img:
            xt = self.transform(img.convert('RGB'))
        
        xt_params = torch.from_numpy(np.load(tgt_param_path).astype(np.float32))
        xt_bbox =  np.load(tgt_box_path)
        xt_bbox = torch.from_numpy(
            np.concatenate([self.fix_bbox(xt_bbox,tube_box),
                self.get_params_box(xt_params.numpy())],-1).astype(np.float32))

        return self.resize(xs),self.resize(xt),xs,xt,xs_params,xt_params,xs_bbox,xt_bbox,flag

    def get_path(self,i):
        idx = i % self.length
        video_paths = self.idinfo[self.keys[idx]]

        if len(video_paths) == 1:
            vIdx = 0 
        else:
            vIdx = random.randint(0, len(video_paths) - 1)
        img_paths = video_paths[vIdx]

        src_idx,tgt_idx = self.select_path(img_paths)
        
        src_img_path = img_paths[src_idx].replace('id','crop').replace('.npy','.png')
        tgt_img_path = img_paths[tgt_idx].replace('id','crop').replace('.npy','.png')

        src_param_path = img_paths[src_idx].replace('id','3dmm')
        tgt_param_path = img_paths[tgt_idx].replace('id','3dmm')

        src_box_path = img_paths[src_idx].replace('id','bbox')
        tgt_box_path = img_paths[tgt_idx].replace('id','bbox')
        return src_img_path,tgt_img_path,src_param_path,tgt_param_path,src_box_path,tgt_box_path

    def get_another_tgt(self,i):
        idx = (i + random.randint(0,self.length-1)) % self.length
        video_paths = self.idinfo[self.keys[idx]]

        if len(video_paths) == 1:
            vIdx = 0 
        else:
            vIdx = random.randint(0, len(video_paths) - 1)
        img_paths = video_paths[vIdx]

        tgt_idx = random.randint(0,len(img_paths)-1)
        
        tgt_img_path = img_paths[tgt_idx].replace('id','crop').replace('.npy','.png')

        tgt_param_path = img_paths[tgt_idx].replace('id','3dmm')

        tgt_box_path = img_paths[tgt_idx].replace('id','bbox')
        return tgt_img_path,tgt_param_path,tgt_box_path

    def fix_bbox(self,bbox,tube_bbox):
        x_min,y_min,x_max,y_max = tube_bbox[:4]
        
        center_x = (x_min + x_max) / 2.0
        center_y = (y_min + y_max) / 2.0
        bbox_size = int(max(y_max-y_min,x_max-x_min) * 1.8)

        x_min = int(center_x-bbox_size * 0.5)
        y_min = int(center_y-bbox_size * 0.5)
        scale = self.size * 1. / bbox_size

        return np.array([(bbox[0] - x_min) * scale,
                        (bbox[1] - y_min) * scale,
                        (bbox[2] - x_min) * scale,
                        (bbox[3] - y_min) * scale]) 

    def select_path(self,img_paths):
        length = len(img_paths)
        if length <= 15:
            src_idx,tgt_idx = 0,-1
        else:
            src_idx = random.randint(0, length - 15-1)
            tgt_idx = random.randint(min(src_idx+15,length-1),length-1)
        return src_idx,tgt_idx

    def get_params_box(self,params):
      
        s,t0,t1 = params.reshape(-1)[-3:]
        s = s + 1e-8
        w = (self.params_w0*s)
        h = (self.params_h0*s)
        
        left = max(0,w/2 - self.params_target_size/2 + float((t0 - self.params_w0/2)*s))
        right = left + self.params_target_size
        up = max(0,h/2 - self.params_target_size/2 + float((self.params_h0/2 - t1)*s))
        below = up + self.params_target_size

        return np.array([left/s,up/s,right/s,below/s])


    def __len__(self):
        if self.eval:
            return max(self.length,1000)
        else:
            # return self.length
            return max(self.length,100000)

