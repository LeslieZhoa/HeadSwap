'''
@author LeslieZhao
@date 20220823
'''
import torch 
from torch import nn 
from model.BlendModule.module import VGG19_pytorch,Decoder
import torch.nn.functional as F
import pdb
import cv2

class Generator(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.feature_ext = VGG19_pytorch()
        self.decoder = Decoder(ic=args.decoder_ic)
        self.dilate = nn.MaxPool2d(kernel_size=args.dilate_kernel, 
                        stride=1, 
                        padding=args.dilate_kernel//2)

        self.phi = nn.Conv2d(in_channels=args.f_in_channels, 
                            out_channels=args.f_inter_channels, kernel_size=1, stride=1, padding=0)
        self.theta = nn.Conv2d(in_channels=args.f_in_channels, 
                            out_channels=args.f_inter_channels, kernel_size=1, stride=1, padding=0)

        self.temperature = args.temperature

        self.head_index = [1,2,3,4,5,6,7,8,9,10,11,12,13,17,18]
        self.eps = 1e-8

    def forward(self,I_a,I_gray,I_t,M_a,M_t,gt=None,cycle=False,train=False):

        fA = self.feature_ext(I_a)
        fT = self.feature_ext(I_t)

        fA = self.phi(fA)
        fT = self.theta(fT)
        
        gen_h,gen_i,mask_list,matrix_list = self.RCNet(fA,fT,M_a,M_t,I_t)

        gen_h = F.adaptive_avg_pool2d(gen_h,I_t.shape[-2:])
        gen_i = F.adaptive_avg_pool2d(gen_i,I_t.shape[-2:])
       
        M_Ah,M_Th,M_Ad,M_Td,M_Ai,M_Ti,M_Ar,M_Tr = mask_list
       

        if cycle:
           
            cycle_gen_h,cycle_gen_i = self.RCCycle(gen_h+gen_i,[M_Ar,M_Tr,M_Ai,M_Ti],matrix_list,fA.shape)
            # cycle_gen = self.RCCycle(gen_h+gen_i,fA,fT,[M_Ar,M_Tr,M_Ai,M_Ti])

            I_td_h = I_t * M_Th
            I_td_h = F.adaptive_avg_pool2d(I_td_h,cycle_gen_h.shape[-2:])

            I_td_i = I_t * M_Ti
            I_td_i = F.adaptive_avg_pool2d(I_td_i,cycle_gen_i.shape[-2:])


            # cat_img = torch.cat([I_t,gen_h,I_a,F.interpolate(cycle_gen_h,size=I_t.shape[-2:])],-1)
            # cv2.imwrite('2.png',(cat_img[0].permute(1,2,0).detach().cpu().numpy()[...,::-1]+1)*127.5)
          
            return cycle_gen_h,cycle_gen_i,I_td_h,I_td_i,gen_h+gen_i
        I_tb = gt * (1-M_Ad)
        I_ag = I_gray * M_Ah

        inp = torch.cat([gen_h,gen_i,
                    M_Ah,
                    I_tb,M_Ai,I_ag],1)
        
        oup = self.decoder(inp)

        if train:
            return oup,M_Ah,M_Ai,gen_h+gen_i

        return oup
       

    def RCNet(self,fA,fT,M_a,M_t,I_t):
        
        M_Ah = self.get_mask(M_a,self.head_index)
        M_Th = self.get_mask(M_t,self.head_index)

        M_Ti,M_Td = self.get_inpainting(M_Th)
        M_Ai,M_Ad = self.get_inpainting(M_Ah+M_Th,M_Ah)
        M_Ar = self.get_multi_mask(M_a)
        M_Tr = self.get_multi_mask(M_t)

        matrix_list = []
        gen_h = None
        for m_a,m_t in zip(M_Ar,M_Tr):
            gen_h, matrix = self.compute_corre(fA,fT,m_a,m_t,I_t,gen_h)
            matrix_list.append(matrix)
       
        gen_i = None 
        gen_i,matrix = self.compute_corre(fA,fT,M_Ai,M_Ti,I_t,gen_i)
        matrix_list.append(matrix)

        
        return gen_h,gen_i,[M_Ah,M_Th,M_Ad,M_Td,M_Ai,M_Ti,M_Ar,M_Tr],matrix_list

    # def RCCycle(self,I_ta,fA,fT,mask_list):
    #     M_Ar,M_Tr,M_Ai,M_Ti = mask_list
    #     gen_h = None
    #     for m_a,m_t in zip(M_Ar,M_Tr):
    #         gen_h, matrix = self.compute_corre(fT,fA,m_t,m_a,I_ta,gen_h)
           
       
    #     gen_i = None 
    #     gen_i,matrix = self.compute_corre(fT,fA,M_Ti,M_Ai,I_ta,gen_i)
        

    #     return gen_h + gen_i


    def RCCycle(self,I_t,mask_list,matrix_list,shape):
        M_Ar,M_Tr,M_Ai,M_Ti = mask_list
        batch,channel,h,w = shape
        gen_h = torch.zeros((batch,3,h,w)).to(I_t.device)
        gen_i = torch.zeros((batch,3,h,w)).to(I_t.device)
        
        I_t_resize = F.adaptive_avg_pool2d(I_t,[h,w])
       
        M_Tr_resize = [F.interpolate(f, size=(h,w),mode='nearest') for f in M_Tr]
        M_Ar_resize = [F.interpolate(f, size=(h,w),mode='nearest') for f in M_Ar]
        M_Ti_resize = F.interpolate(M_Ti, size=(h,w),mode='nearest')
        M_Ai_resize = F.interpolate(M_Ai, size=(h,w),mode='nearest') 

       
        for matrix,m_t,m_a in zip(matrix_list[:-1],M_Tr_resize,M_Ar_resize):
            for i in range(batch):
                f_WTA = matrix[i]
                f = F.softmax(f_WTA.transpose(1,2),dim=-1)
                
                ref = torch.matmul(
                    I_t_resize[i].unsqueeze(0).masked_select(
                        m_a[i].unsqueeze(0)==1).view(1,3,-1),f.transpose(1,2))
                gen_h[i] = gen_h[i].unsqueeze(0).masked_scatter(
                        m_t[i].unsqueeze(0)==1,ref).squeeze(0)


        for i in range(batch):

            f_WTA = matrix_list[-1][i]
            f = F.softmax(f_WTA.transpose(1,2),dim=-1)
           
            ref = torch.matmul(
                I_t_resize[i].unsqueeze(0).masked_select(
                    M_Ai_resize[i].unsqueeze(0)==1).view(1,3,-1),f.transpose(1,2))
            gen_i[i] = gen_i[i].unsqueeze(0).masked_scatter(
                    M_Ti_resize[i].unsqueeze(0)==1,ref).squeeze(0)

       

        return gen_h,gen_i

    def compute_corre(self,fA,fT,M_A,M_T,I_t,gen=None):
        batch,channel,h,w = fA.shape
        matrix_list = []
        if gen is None:
            gen = torch.zeros((batch,3,h,w)).to(I_t.device)
        
        M_A_resize = F.interpolate(M_A, size=(h,w),mode='nearest')
        M_T_resize = F.interpolate(M_T, size=(h,w),mode='nearest')
        I_t_resize = F.adaptive_avg_pool2d(I_t,[h,w])

        for i in range(batch):
            fAr = fA[i].unsqueeze(0).masked_select(
                M_A_resize[i].unsqueeze(0)==1).view(1,channel,-1) # b,c,hA

            fTr = fT[i].unsqueeze(0).masked_select(
                M_T_resize[i].unsqueeze(0)==1).view(1,channel,-1) # b,c,hT

            fAr = self.normlize(fAr)
            fTr = self.normlize(fTr)
    
            matrix = torch.matmul(fAr.permute(0,2,1),fTr) # b,hA,hT
            f_WTA = matrix/self.temperature
            f = F.softmax(f_WTA,dim=-1)
            matrix_list.append(f_WTA)

            ref = torch.matmul(
                I_t_resize[i].unsqueeze(0).masked_select(
                M_T_resize[i].unsqueeze(0)==1).view(1,3,-1),f.transpose(1,2)) # [b,channel,hT] X [b,hT,hA]
            
            
            gen[i] = gen[i].unsqueeze(0).masked_scatter(
                    M_A_resize[i].unsqueeze(0)==1,ref).squeeze(0)
            
       
        return gen,matrix_list

    def get_inpainting(self,M,head=None):
        M = torch.clamp(M,0,1)
        M_dilate = self.dilate(M)
        if head is None:
            MI = M_dilate - M
        else:
            MI = M_dilate - head
        return MI,M_dilate
    
    def get_multi_mask(self,M_a):
        # skin
        skin_mask_A = self.get_mask(M_a,[1])
        # hair 
        hair_mask_A = self.get_mask(M_a,[17,18])

        # eye 
        eye_mask_A = self.get_mask(M_a,[4,5,6])

        # brow
        brow_mask_A = self.get_mask(M_a,[2,3])

        # ear 
        ear_mask_A = self.get_mask(M_a,[7,8,9])

        #nose
        nose_mask_A = self.get_mask(M_a,[10])

        # lip
        lip_mask_A = self.get_mask(M_a,[12,13])


        # tooth
        tooth_mask_A = self.get_mask(M_a,[11])
        
        return [skin_mask_A,hair_mask_A,eye_mask_A,brow_mask_A,ear_mask_A,nose_mask_A,lip_mask_A,tooth_mask_A]

    def get_mask(self,mask,indexs):
        out = torch.zeros_like(mask)
        for i in indexs:
            out[mask==i] = 1

        return out

    def normlize(self,x):
        x = x - x.mean(dim=1,keepdim=True)
        x_norm = torch.norm(x,2,1,keepdim=True) + self.eps 
        return x / x_norm
        # return x
