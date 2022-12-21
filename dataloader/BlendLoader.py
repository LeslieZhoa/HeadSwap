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
import torchvision.transforms.functional as F
from dataloader.augmentation import ParametricAugmenter
import math
import numpy as np
import cv2


class BlendData(DatasetBase):
    def __init__(self, slice_id=0, slice_count=1,dist=False, **kwargs):
        super().__init__(slice_id, slice_count,dist, **kwargs)


        self.transform = transforms.Compose([
            transforms.Resize((kwargs['size'], kwargs['size'])),
            transforms.ToTensor()
        ])
        self.color_fn2 = transforms.Compose([transforms.ColorJitter(0.5, 0.5, 0.5, 0.1)])
        
        
        self.norm = transforms.Compose([transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])])

        self.gray = transforms.Compose([transforms.Grayscale(num_output_channels=1)])

        self.aug_fn = ParametricAugmenter(use_pixelwise_augs=False,
                                        use_affine_scale=kwargs['use_affine_scale'],
                                        use_affine_shift=kwargs['use_affine_shift'])

        self.color_fn = ParametricAugmenter(use_pixelwise_augs=True,
                                        use_affine_scale=False,
                                        use_affine_shift=False)

        # source root
        root = kwargs['root']
        self.idinfo = np.load(root,allow_pickle=True).item()
        keys = list(self.idinfo.keys())
        
        dis = math.floor(len(keys)/self.count)
        self.keys = keys[self.id*dis:(self.id+1)*dis]
        self.length = len(self.keys)
        random.shuffle(self.keys)

        # landscope
        landscope_root = kwargs['landscope_root']
        self.landscope_paths = [os.path.join(landscope_root,f) for f in os.listdir(landscope_root)]
        self.landscope_length = len(self.landscope_paths)
        random.shuffle(self.landscope_paths)

        # Fabric
        fabric_root = kwargs['fabric_root']
        self.fabric_paths = [os.path.join(fabric_root,f) for f in os.listdir(fabric_root)]
        self.fabric_length = len(self.fabric_paths)
        random.shuffle(self.fabric_paths)
        self.eval = kwargs['eval']
        
       

    def __getitem__(self,i):
        
        idx = i % self.length
        img_path,mask_path = self.get_img_path(idx)

        idx = (i + random.randint(0,self.length-1)) % self.length
        
        ex_img_path,ex_mask_path = self.get_img_path(idx)

        # landscope
        landscope_path = self.landscope_paths[random.randint(0, self.landscope_length - 1)]

        # fabric
        fabric_path = self.fabric_paths[random.randint(0, self.fabric_length - 1)]

        gt = cv2.imread(img_path)
        M_a = cv2.imread(mask_path)
        landscope = cv2.imread(landscope_path)
        fabric = cv2.imread(fabric_path)

        if random.random() > 0.5 and not self.eval:
            # change clothes
            I_a = self.change_clothes(gt,M_a,landscope,fabric)
        else:
            I_a = gt.copy()

        I_a,M_a,gt = self.numpy2img(I_a,M_a,gt)
        

        
        gt = self.transform(gt.convert('RGB'))
        M_a = self.transform(M_a.convert('L'))  
        I_a = self.transform(I_a.convert('RGB'))
        I_a = self.color_fn2(self.color_fn.augment_tensor(I_a))
        
        I_a = self.norm(I_a)
        I_gray = self.gray(I_a)
        I_t,M_t = self.aug_fn.augment_double(gt,M_a)
        if random.random() > 0.3:
            I_t = F.hflip(I_t)
            M_t = F.hflip(M_t)
        
       
        gt = self.norm(gt)
        I_t = self.norm(I_t)
        M_a = M_a * 255
        M_t = M_t * 255
        
        with Image.open(ex_img_path) as img:
            hat_t = self.transform(img.convert('RGB'))
        hat_t = self.norm(hat_t)
        with Image.open(ex_mask_path) as img:
            M_hat = self.transform(img.convert('L')) * 255

        return I_a,I_gray,I_t,hat_t,M_a,M_t,M_hat,gt

    def get_img_path(self,idx):
        video_paths = self.idinfo[self.keys[idx]]

        if len(video_paths) == 1:
            vIdx = 0 
        else:
            vIdx = random.randint(0, len(video_paths) - 1)
        img_paths = video_paths[vIdx]
        img_idx = random.randint(0, len(img_paths) - 1)
        img_path = img_paths[img_idx].replace('id','crop').replace('.npy','.png')
        mask_path = img_path.replace('crop','mask')

        return img_path,mask_path

    def change_clothes(self,gt,M_a,landscope,fabric):
        
        h,w,_ = gt.shape
        fabric = cv2.resize(fabric,[w,h])
        hl,wl,_ = landscope.shape
        scale = max(1,max(h*1./hl,w*1./wl))
        landscope = cv2.resize(landscope,None,fx=scale,fy=scale)
        hl,wl,_ = landscope.shape
        left = random.randint(0,max(0,wl-w-1))
        top = random.randint(0,max(0,hl-h-1))
        landscope = landscope[top:top+h,left:left+w]

        fuse = np.where(M_a==0,landscope,gt)
        fuse = np.where(M_a==16,fabric,fuse)
        return fuse

    def numpy2img(self,*args):
        outputs = []
        for arg in args:
            outputs.append(Image.fromarray(cv2.cvtColor(arg,cv2.COLOR_BGR2RGB)))
        return outputs

    
    def __len__(self):
        if self.eval:
            return max(self.length,1000)
            # return 10
        else:
            # return self.length
            return max(self.length,100000)


