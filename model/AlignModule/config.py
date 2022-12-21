class Params:
    def __init__(self):
        self.stage = 'gen'
        self.name = 'Aligner'
        self.pretrain_path = 'checkpoint/Aligner/058-00010900.pth'
        self.size = 512
        self.train_root = 'HeadSwap/wav2lip-headswap/info/train_id.npy'
        self.val_root = 'HeadSwap/wav2lip-headswap/info/val_id.npy'
        self.id_model = 'pretrained_models/model_ir_se50.pth'
        self.params_model = 'pretrained_models/epoch_20.pth'
        self.scratch_path = 'PIRender/result/face/epoch_00190_iteration_000400000_checkpoint.pt'
        

        self.frozen_params = ['mapping_net','warpping_net']
        self.train_params = ['encoder_net','editing_net']
        # MappingNet
        self.m_layer = 3

        # WarpingNet
        self.w_encoder_layer = 5
        self.w_decoder_layer = 3
        self.w_base_nc = 32 

        # EditingNet
        self.e_layer = 3
        self.e_num_res_blocks = 2
        self.e_base_nc = 64

        # Common
        self.image_nc = 3
        self.descriptor_nc = 256
        self.coeff_nc = 73
        self.max_nc = 256
        self.use_spect = False
        self.driving_num = 27

        # Descriminator
        self.d_input_nc = 3

        # Loss 
        self.per_num_scales = 4
        

        self.rec_loss_warp = True 
        self.per_loss_warp = True 
        self.reg_loss = True 
        self.featLoss = True
        self.per_loss_final = True
        self.rec_loss_final = True
        self.id_loss = True
        self.params_loss = True
        self.featLoss = True

        # warp
        # self.lambda_wrec = 100
        # self.lambda_wper = 1
        # self.lambda_reg = 100
        # self.lambda_gan = 2
        # self.lambda_feat = 1
        # self.lambda_gper = 1
        # self.lambda_grec = 1
        # self.lambda_id = 1
        # self.lambda_params = 50
        # self.lambda_params_gen = 50

        # gen
        # self.lambda_wrec = 80
        # self.lambda_wper = 0.5
        # self.lambda_params = 10
        # self.lambda_reg = 100
        # self.lambda_gan = 20
        # self.lambda_feat = 25
        # self.lambda_gper = 1
        # self.weight_style_to_perceptual = 250
        # self.lambda_grec = 100
        # self.lambda_id = 8
        
        # self.lambda_params_gen = 50

        self.lambda_wrec = 80
        self.lambda_wper = 0.6
        self.lambda_params = 10
        self.lambda_reg = 100
        self.lambda_gan = 10
        self.lambda_feat = 100
        self.lambda_gper = 1
        self.weight_style_to_perceptual = 1000
        self.lambda_grec = 150
        self.lambda_id = 30
        
        self.lambda_params_gen = 50

        self.g_lr = 1e-4
        self.d_lr = 4e-5
        self.beta1 = 0.9
        self.beta2 = 0.999
