import torch 
from PIL import Image
from .network import ReconNetWrapper
from .preprocess import align_img,load_lm3d
import face_alignment
import numpy as np
import pdb
class FaceRec:
    def __init__(self,model_path,bfm_folder,use_lmk=False):
        self.model = ReconNetWrapper()
        self.model.to('cuda')
        self.model.load_state_dict(torch.load(model_path)['net_recon'])
        self.model.eval()
        self.lm3d_std = load_lm3d(bfm_folder) 
        if use_lmk:
            self.lmk_detector = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D)   

    def run(self,x,lmk=None):
        img,lm,trans_params = self.preprocess(x,lmk)
        params = self.forward(img.cuda())
        y = self.postprocess(params,trans_params)
        return y
        
    def forward(self,x):
        with torch.no_grad():
            y = self.model(x)
        return y

    def preprocess(self,x,scale=1,lmk=None):
        
        if lmk is None:
            lmk = self.lmk_detector.get_landmarks_from_image(x)[0]
        
        img, lm, trans_params = self.image_transform(Image.fromarray(x),lmk)
        return img,lm,trans_params

    def postprocess(self,coeff_3dmm,crop_param):
        
        coeff_3dmm = coeff_3dmm.cpu().numpy()
        # id_coeff = coeff_3dmm[:,:80] #identity
        ex_coeff = coeff_3dmm[:,80:144] #expression
        # tex_coeff = coeff_3dmm[:,144:224] #texture
        angles = coeff_3dmm[:,224:227] #euler angles for pose
        # gamma = coeff_3dmm[:,227:254] #lighting
        translation = coeff_3dmm[:,254:257] #translation
    
        coeff_3dmm = np.concatenate([ex_coeff, angles, translation, crop_param.reshape(1,-1)], 1)
        return coeff_3dmm.transpose()

    def image_transform(self, images, lm):
        W,H = images.size
        
        lm[:, -1] = H - 1 - lm[:, -1]

        trans_params, img, lm, _ = align_img(images, lm, self.lm3d_std)        
        img = torch.tensor(np.array(img)/255., dtype=torch.float32).permute(2, 0, 1)
        _, _, ratio, t0, t1 = np.hsplit(trans_params.astype(np.float32), 5)
        trans_params = np.concatenate([ratio, t0, t1], 0)
        return img.unsqueeze(0), lm, trans_params   

    

    