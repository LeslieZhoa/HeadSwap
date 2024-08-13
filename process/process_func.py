import sys 
sys.path.append('.')
sys.path.append('..')
from face_alignment.detection.sfd import FaceDetector
import face_alignment
from process.process_utils import *
import torch
import numpy as np
from model.third.Deep3dRec.network import ReconNetWrapper
from model.third.Deep3dRec.preprocess import align_img,load_lm3d
from PIL import Image

class Process:
    def __init__(self,params_path,bfm_folder):
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = 'cuda'
        self.face_detector = FaceDetector(device='cuda')
        self.lmk_detector = face_alignment.FaceAlignment(face_alignment.LandmarksType.THREE_D, flip_input=False)
        # 3dmm params
        self.ParamsModel = ReconNetWrapper()
        self.ParamsModel.to(self.device)
        self.ParamsModel.load_state_dict(torch.load(params_path)['net_recon'])
        self.lm3d_std = load_lm3d(bfm_folder) 
        self.mean =torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
        self.std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)

    def preprocess_align(self,img,size=512,top_scale=0.5):
        bboxes = self.detect_faces([img])[0]
            
        bboxes = list(filter(lambda x:x[-1]>0.99,bboxes))

        if len(bboxes) == 0:
            return None,None 

        bbox = bboxes[0]
        
        landmarks = self.lmk_detector.get_landmarks_from_image(img[...,::-1], [bbox])[0]
        image_cropped,info = crop_with_padding(img,landmarks[:,:2],bbox,scale=1.8,size=size,align=True,top_scale=top_scale)

        return image_cropped,info

    def get_params(self,img,lmk):
        
        img,_,crop_param = self.preprocess_params(img,lmk) 
        with torch.no_grad():
            coeff_3dmm = self.ParamsModel(img)
       
        ex_coeff = coeff_3dmm[:,80:144] #expression
      
        angles = coeff_3dmm[:,224:227] #euler angles for pose
     
        translation = coeff_3dmm[:,254:257] #translation

        coeff_3dmm = torch.cat([ex_coeff, angles, translation, crop_param.view(1,-1)], 1)
        return coeff_3dmm.permute(1,0)

    def preprocess(self,x,size=512):
        if isinstance(x,str):
            x = cv2.imread(x)
        x = cv2.resize(x,[size,size])
        x = (x[...,::-1].transpose(2,0,1)[np.newaxis,:] / 255 - 0.5) * 2
      
        return torch.from_numpy(x.astype(np.float32)).to(self.device)

    def preprocess_params(self,img,lm):
    
        images = Image.fromarray(img)
        W,H = images.size
        
        lm[:, -1] = H - 1 - lm[:, -1]

        trans_params, img, lm, _ = align_img(images, lm, self.lm3d_std)        
        img = torch.tensor(np.array(img)/255., dtype=torch.float32).permute(2, 0, 1)
        _, _, ratio, t0, t1 = np.hsplit(trans_params.astype(np.float32), 5)
        trans_params = torch.tensor(np.concatenate([ratio, t0, t1], 0))
        return img.unsqueeze(0).to(self.device), lm, trans_params.to(self.device)

    def preprocess_parsing(self,x):
        
        return ((x+1)/2.0 - self.mean.view(1,-1,1,1).to(self.device)) / \
                self.std.view(1,-1,1,1).to(self.device)


    def postprocess(self,x):
        return (x.permute(1,2,0).cpu().numpy()[...,::-1] + 1) * 127.5

    

    def postprocess_parsing(self,x):
       
        return torch.argmax(x[0],1).unsqueeze(1).float()

    def detect_faces(self,images):
        images = np.stack(images).transpose(0,3,1,2).astype(np.float32)
        images_torch = torch.tensor(images)
        return self.face_detector.detect_from_batch(images_torch.cuda())


if __name__ == "__main__":
    base = 'test_img'
    save = 'test-img-crop'
    import os

    os.makedirs(save,exist_ok=True)
    model = Process('pretrained_models/epoch_20.pth',
                'pretrained_models/BFM')
    i = 0
    for name in os.listdir(base):
        img_path = os.path.join(base,name)
        img = cv2.imread(img_path)
        img_crop,info = model.preprocess_align(img)
        params = model.get_params(cv2.resize(img_crop,(256,256)),
                                info['rotated_lmk']/2.0)
        cv2.imwrite(os.path.join(save,name),img_crop)
        
        np.save(os.path.join(save,os.path.splitext(name)[0]),params.cpu().numpy())
        print('\rhave done %04d'%i,end='',flush=True)
        i += 1
    print()
        
