import torch.nn as nn
from model.AlignModule.lib import *
class FaceGenerator(nn.Module):
    def __init__(
        self, 
        args
        ):  
        super(FaceGenerator, self).__init__()

        self.args = args
        # self.encoder_net = EncoderNet(args.coeff_nc)
        self.mapping_net = MappingNet(args.coeff_nc, 
                                      args.descriptor_nc, 
                                      args.m_layer)
        self.warpping_net = WarpingNet(args.image_nc, 
                    args.descriptor_nc, 
                    args.w_base_nc, 
                    args.max_nc, 
                    args.w_encoder_layer, 
                    args.w_decoder_layer, 
                    args.use_spect)
        self.editing_net = EditingNet(args.image_nc, 
                    args.descriptor_nc, 
                    args.e_layer, 
                    args.e_base_nc, 
                    args.max_nc, 
                    args.e_num_res_blocks, 
                    args.use_spect)
 
    def forward(
        self, 
        input_image,
        tgt_image, 
        driving_source, 
        stage=None
        ):
        
        # error = self.encoder_net(tgt_image)
       
        # fix_driving = driving_source+error.unsqueeze(-1)
        # descriptor = self.mapping_net(fix_driving.repeat(1,1,self.args.driving_num))
        descriptor = self.mapping_net(driving_source.repeat(1,1,self.args.driving_num))
        output = self.warpping_net(input_image, descriptor)
        # output['error'] = error
        
        if stage != 'warp':
            output['fake_image'] = self.editing_net(input_image, output['warp_image'], descriptor)
        return output


