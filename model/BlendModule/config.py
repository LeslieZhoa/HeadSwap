'''
@author LeslieZhao
@date 20221221
'''
class Params:
    def __init__(self):
       
        self.name = 'Blender'
        self.pretrain_path = 'checkpoint/Blender/400-00005400.pth'
        self.size = 512

        self.train_root = 'HeadSwap/wav2lip-headswap/info/train_id.npy'
        self.val_root = 'HeadSwap/wav2lip-headswap/info/val_id.npy'
        self.landscope_root = 'HeadSwap/landscope'
        self.fabric_root = 'HeadSwap/cropfix'

        self.f_in_channels = 512
        self.f_inter_channels = 256
        self.temperature = 0.001
        self.dilate_kernel = 17
        self.decoder_ic = 12

        # discriminator
        
        self.d_input_nc = 5
        
        
        
        # loss
        self.per_num_scales = 4
        self.featLoss = True
        self.rec_loss = True 
        self.per_loss = True 
        self.featLoss = True
        self.lambda_gan = 4
        self.lambda_cycle = 20
        self.lambda_rec = 100.0
        self.lambda_per = 0.4
        self.lambda_feat = 5.0

        self.g_lr = 1e-3
        self.d_lr = 4e-3
        self.beta1 = 0.9
        self.beta2 = 0.999

        self.use_affine_scale = True
        self.use_affine_shift = True