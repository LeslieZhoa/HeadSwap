from model.AlignModule.generator import FaceGenerator
from model.BlendModule.generator import Generator as Decoder
from model.AlignModule.config import Params as AlignParams
from model.BlendModule.config import Params as BlendParams 
from model.third.faceParsing.model import BiSeNet
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import torch
import cv2
import numpy as np
import pdb
from process.process_func import Process
from process.process_utils import *
import os
import onnxruntime as ort
from utils.utils import color_transfer2

class Infer(Process):
    def __init__(self,align_path,blend_path,parsing_path,params_path,bfm_folder):
        Process.__init__(self,params_path,bfm_folder)
        align_params = AlignParams()
        blend_params = BlendParams()
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = 'cuda'
      
        self.parsing = BiSeNet(n_classes=19).to(self.device)
        
        self.netG = FaceGenerator(align_params).to(self.device)
        self.decoder = Decoder(blend_params).to(self.device)
        
        self.loadModel(align_path,blend_path,parsing_path)
        self.eval_model(self.netG,self.decoder,self.parsing)
        

        self.ort_session_sr = ort.InferenceSession('./pretrained_models/sr_cf.onnx', providers=['CPUExecutionProvider'])

    def run(self,src_img_path_list,tgt_img_path_list,save_base,crop_align=False,cat=False):
        os.makedirs(save_base,exist_ok=True)
        i = 0
        for src_img_path,tgt_img_path in zip(src_img_path_list,tgt_img_path_list):
            gen = self.run_single(src_img_path,tgt_img_path,crop_align=crop_align,cat=cat)
            img_name = os.path.splitext(os.path.basename(src_img_path))[0]+'-' + \
                        os.path.splitext(os.path.basename(tgt_img_path))[0]+'.png'
            cv2.imwrite(os.path.join(save_base,img_name),gen)
            print('\rhave done %04d'%i,end='',flush=True)
            i += 1
        print()
    def run_single(self,src_img_path,tgt_img_path,crop_align=False,cat=False):
        
        tgt_img = cv2.imread(tgt_img_path)
        tgt_align = tgt_img.copy()
        
        tgt_align,info = self.preprocess_align(tgt_img)
        if tgt_align is None:
            return None

        src_img = cv2.imread(src_img_path)
        src_align = src_img
        if crop_align:
            src_align,_ = self.preprocess_align(src_img,top_scale=0.55)
        
        src_inp = self.preprocess(src_align)
        tgt_inp = self.preprocess(tgt_align)

        tgt_params = self.get_params(cv2.resize(tgt_align,(256,256)),
                                info['rotated_lmk']/2.0).unsqueeze(0)
           
        gen = self.forward(src_inp,tgt_inp,tgt_params) 

        gen = self.postprocess(gen[0])
        gen = self.run_sr(gen)
        mask = self.mask
        final = gen
        # gen = color_transfer2(tgt_align,gen)
            
        RotateMatrix = info['im'][:2]
        mask = info['mask'][...,0]
       
        rotate_gen = cv2.warpAffine(gen, RotateMatrix, (tgt_img.shape[1], tgt_img.shape[0]))
        mask = cv2.warpAffine(mask, RotateMatrix, (tgt_img.shape[1], tgt_img.shape[0])) * 1.0

        # ori_mask = mask.copy()
        kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT,(17, 17))
        # mask = cv2.dilate(mask*1.0,kernel2)
        mask = cv2.erode(mask*1.0,kernel2)
        # mask = cv2.GaussianBlur(mask*255.0, (21, 21), 0) / 255.0
        mask = cv2.blur(mask*1.0, (15, 15), 0) / 255.0
        mask = np.clip(mask,0,1.0)[:,:,np.newaxis]

        # pdb.set_trace()
        final = rotate_gen * mask + tgt_img * (1-mask)

        if cat:
            final = np.concatenate([tgt_img,final],1)
            final[-256:,:256] = cv2.resize(src_align,(256,256))

        return final
    
    def forward(self,xs,xt,params):
        with torch.no_grad():

            # xg = self.netG(F.adaptive_avg_pool2d(xs,256),
            #                 F.adaptive_avg_pool2d(xt,256),
            #                 params)['fake_image']
            xg = F.adaptive_avg_pool2d(self.netG(F.adaptive_avg_pool2d(xs,256),
                            F.adaptive_avg_pool2d(xt,256),
                            params)['fake_image'],512)
           
            
            M_a = self.parsing(self.preprocess_parsing(xg))
           
            M_t = self.parsing(self.preprocess_parsing(xt))
            
            M_a = self.postprocess_parsing(M_a)
            M_t = self.postprocess_parsing(M_t)
            # xg[M_a.repeat(1,3,1,1)==0] = -0.5
            # xg[M_a.repeat(1,3,1,1)==16] = 0.6
            xg_gray = TF.rgb_to_grayscale(xg,num_output_channels=1)
            fake = self.decoder(xg,xg_gray,xt,M_a,M_t,xt,train=False)
            
            
            gen_mask = self.parsing(self.preprocess_parsing(fake))
            gen_mask = self.postprocess_parsing(gen_mask)
            gen_mask = gen_mask[0][0].cpu().numpy()
            mask_t = M_t[0][0].cpu().numpy()
            mask = np.zeros_like(gen_mask)
            for i in [1,2,3,4,5,6,7,8,9,10,11,12,13,17,18]:
                mask[gen_mask==i] = 1.0
                mask[mask_t==i] = 1.0
            
            self.mask = mask
        return fake
    
    def run_sr(self,input_np):
        input_np = cv2.cvtColor(input_np, cv2.COLOR_BGR2RGB)
        # prepare data
        input_np = input_np.transpose((2,0,1))
        input_np = np.array(input_np[np.newaxis, :])
        outputs_onnx = self.ort_session_sr.run(None, {'input_image':input_np.astype(np.uint8)})

        out_put_onnx = outputs_onnx[0]
        outimg = out_put_onnx[0,...].transpose(1,2,0)
        outimg = cv2.cvtColor(outimg, cv2.COLOR_BGR2RGB)
        return outimg

        
    def loadModel(self,align_path,blend_path,parsing_path):
        ckpt = torch.load(align_path, map_location=lambda storage, loc: storage)
        # self.netG.load_state_dict(ckpt['G'])
        self.netG.load_state_dict(ckpt['net_G_ema'])

        ckpt = torch.load(blend_path, map_location=lambda storage, loc: storage)
        self.decoder.load_state_dict(ckpt['G'],strict=False)

        self.parsing.load_state_dict(torch.load(parsing_path))

    
    def eval_model(self,*args):
        for arg in args:
            arg.eval()



if __name__ == "__main__":
    model = Infer(
                # 'checkpoint/Aligner/058-00008100.pth',
                'pretrained_models/epoch_00190_iteration_000400000_checkpoint.pt',
                'pretrained_models/Blender-401-00012900.pth',
                'pretrained_models/parsing.pth',
                'pretrained_models/epoch_20.pth',
                'pretrained_models/BFM')

    # find_path = lambda x: [os.path.join(x,f) for f in os.listdir(x)]
    # img_paths = find_path('../HeadSwap/test_img')[::-1]
    
    src_paths = ['./assets/5.jpg']
    tgt_paths = ['assets/fe54875c-2cf0-4147-b08a-80552a9f46be.jpg']
    
    model.run(src_paths,tgt_paths,save_base='res-1125',crop_align=True,cat=True)
    
   