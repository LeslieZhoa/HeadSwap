#! /usr/bin/python 
# -*- encoding: utf-8 -*-
'''
@author LeslieZhao
@date 20221221
'''
import torch
from trainer.ModelTrainer import ModelTrainer
from model.AlignModule.generator import FaceGenerator
from model.AlignModule.discriminator import Discriminator
from utils.utils import *
import torch.nn.functional as F
from model.AlignModule.criterion import *
import random
import torch.distributed as dist
from itertools import chain
import pdb

class AlignTrainer(ModelTrainer):

    def __init__(self, args):
        super().__init__(args)
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = 'cuda'
      
        self.netG = FaceGenerator(args).to(self.device)
        self.g_ema = FaceGenerator(args).to(self.device)
        self.g_ema.eval()
        self.netD = None
        if args.stage != 'warp':
            self.netD = Discriminator(args.d_input_nc).to(self.device)
            init_weights(self.netD,'xavier_uniform')

        init_weights(self.netG,'xavier_uniform')

        self.optimG,self.optimD = self.create_optimizer() 

        if not self.args.scratch:
            self.load_scratch_path()

        if args.pretrain_path is not None:
            self.loadParameters(args.pretrain_path)

        

        if self.args.frozen_params is not None:
            self.freeze_models()
        accumulate(self.g_ema,self.netG,0)
        if args.dist:
            self.netG,self.netG_module = self.use_ddp(self.netG)
            self.netD,self.netD_module = self.use_ddp(self.netD)
        else:
            self.netG_module = self.netG 
            self.netD_module = self.netD
        
        if self.args.per_loss_warp:
            self.perWarpLoss = PerceptualLoss(
                                use_style_loss=False,
                                num_scales=args.per_num_scales).to(self.device)
            self.perWarpLoss.eval()

        if self.args.per_loss_final:
            self.perFinalLoss = PerceptualLoss(
                                use_style_loss=True,
                                weight_style_to_perceptual=args.weight_style_to_perceptual,
                                num_scales=args.per_num_scales).to(self.device)
            self.perFinalLoss.eval()

        if self.args.id_loss:
            self.idLoss = IDLoss(args.id_model).to(self.device)
            self.idLoss.eval()
        
        if self.args.params_loss:
            self.paramsLoss = ParamsLoss(args.params_model).to(self.device)
        
        self.L1Loss = torch.nn.L1Loss()
        self.accum = 0.5 ** (32 / (10 * 1000))
        

    def create_optimizer(self):
       
        if self.args.train_params is None:
            g_params = self.netG.parameters()
        else:
            g_params = chain(*[getattr(self.netG,key).parameters() for key in self.args.train_params])
        g_optim = torch.optim.Adam(
                    g_params,
                    lr=self.args.g_lr,
                    betas=(self.args.beta1, self.args.beta2),
                    )
        d_optim = None
        if self.netD is not None:
            d_optim = torch.optim.Adam(
                        self.netD.parameters(),
                        lr=self.args.d_lr,
                        betas=(self.args.beta1, self.args.beta2),
                        )
        
        return  g_optim,d_optim

    
    def run_single_step(self, data, steps):
        self.netG.train()
        data = self.process_input(data)
        if self.netD is not None:
            self.run_discriminator_one_step(data,steps)
        self.run_generator_one_step(data,steps)
        

    def run_discriminator_one_step(self, data,step):
        
        D_losses = {}
        requires_grad(self.netG, False)
        requires_grad(self.netD, True)

        xs_resize,xt_resize,xs,xt,xs_params,xt_params,xs_bbox,xt_bbox,flag = data 
        xs_inp = torch.cat([xs_resize,xt_resize],0)
        xt_inp = torch.cat([xt_resize,xs_resize],0)
        params = torch.cat([xt_params,xs_params],0)
        gt = torch.cat([xt,xs],0)
        xg = self.netG(xs_inp,xt_inp,params)['fake_image']
        fake_pred  = self.netD(xg)
        real_pred  = self.netD(gt)
        d_loss = compute_dis_loss(fake_pred[-1],real_pred[-1],D_losses)

        D_losses['d'] = d_loss
        
        self.optimD.zero_grad()
        d_loss.backward()
        self.optimD.step()
        
        self.d_losses = D_losses


    def run_generator_one_step(self, data,step):
        
        
        requires_grad(self.netG, True)
        requires_grad(self.netD, False)
        
        xs_resize,xt_resize,xs,xt,xs_params,xt_params,xs_bbox,xt_bbox,flag = data 
        G_losses,loss,xg = \
            self.compute_g_loss(xs_resize,xt_resize,xs,xt,xs_params,xt_params,xs_bbox,xt_bbox,flag)

        self.optimG.zero_grad()
        loss.mean().backward()
        self.optimG.step()
        accumulate(self.g_ema,self.netG_module,self.accum)

        self.g_losses = G_losses
        batch_size = xs.shape[0]
        if self.netD is None:
            self.generator = [xs_resize.detach(),xg[:batch_size].detach(),xt_resize.detach()]
        else:

            self.generator = [xs.detach(),xg[:batch_size].detach(),xt.detach()]
        
    
    def evalution(self,test_loader,steps,epoch):
        
        loss_dict = {}
        index = random.randint(0,len(test_loader)-1)
        counter = 0
        with torch.no_grad():
            for i,data in enumerate(test_loader):
                
                data = self.process_input(data)
                xs_resize,xt_resize,xs,xt,xs_params,xt_params,xs_bbox,xt_bbox,flag = data 
                G_losses,loss,xg = \
                    self.compute_g_loss(xs_resize,xt_resize,xs,xt,xs_params,xt_params,xs_bbox,xt_bbox,flag)
    
                for k,v in G_losses.items():
                    loss_dict[k] = loss_dict.get(k,0) + v.detach()
                if i == index and self.args.rank == 0 :
                    ema_oup = self.g_ema(xs_resize,xt_resize,xt_params,stage=self.args.stage)
                    batch_size = xs.shape[0]
                    if self.args.stage == 'warp':
                        show_data = [xs_resize.detach(),
                                    xg[:batch_size].detach(),
                                    ema_oup['warp_image'].detach(),
                                    xt_resize.detach()]

                    else:
                        show_data = [xs.detach(),
                                        xg[:batch_size].detach(),
                                        ema_oup['fake_image'].detach(),
                                        xt.detach()]

                    self.val_vis.display_current_results(self.select_img(show_data,size=show_data[-1].shape[-1]),steps)
                counter += 1
        
       
        for key,val in loss_dict.items():
            loss_dict[key] /= counter

        if self.args.dist:
            # if self.args.rank == 0 :
            dist_losses = loss_dict.copy()
            for key,val in loss_dict.items():
                
                dist.reduce(dist_losses[key],0)
                value = dist_losses[key].item()
                loss_dict[key] = value / self.args.world_size

        if self.args.rank == 0 :
            self.val_vis.plot_current_errors(loss_dict,steps)
            self.val_vis.print_current_errors(epoch+1,0,loss_dict,0)

        return loss_dict
       

    def compute_g_loss(self,xs_resize,xt_resize,xs,xt,xs_params,xt_params,xs_bbox,xt_bbox,flag):

       
        G_losses = {}
        loss = 0

        xs_inp = torch.cat([xs_resize,xt_resize],0)
        xt_inp = torch.cat([xt_resize,xs_resize],0)
        params = torch.cat([xt_params,xs_params],0)
        bbox_inp = torch.cat([xt_bbox,xs_bbox],0)
        gt = torch.cat([xt,xs],0)
        flag = torch.cat([flag,flag],0).view(-1,1,1,1)
        batch_size = xs_inp.shape[0]

        output = self.netG(xs_inp,xt_inp,params,stage=self.args.stage)
        xw = output['warp_image']

        # warp mode
        if self.args.rec_loss_warp:
            warprec_loss = self.L1Loss(xw*flag,xt_inp*flag) *\
                     batch_size / (flag.sum() + 1e-9)  * self.args.lambda_wrec
            G_losses['warp/rec_loss'] = warprec_loss
            loss += warprec_loss
        
        if self.args.per_loss_warp:
            warppre_loss = self.perWarpLoss(xw*flag,xt_inp*flag) *\
                     batch_size / (flag.sum() + 1e-9) * self.args.lambda_wper
            G_losses['warp/per_loss'] = warppre_loss
            loss += warppre_loss

        if self.args.reg_loss:
            reg_loss = torch.norm(output['error'],p=2,dim=1).mean() * self.args.lambda_reg
            G_losses['warp/reg_loss'] = reg_loss
            loss += reg_loss

        if self.args.params_loss:
            
            params_loss = self.paramsLoss(
                        self.process_id_input(xw,bbox_inp[:,-4:],224),
                        self.process_id_input(xt_inp,bbox_inp[:,-4:],224)

            )  * self.args.lambda_params
            
            G_losses['warp/params_loss'] = params_loss
            loss += params_loss



        # gen mode
        if self.netD is not None:
            xg = output['fake_image']
            fake_pred = self.netD(xg)

            # gan loss
            gan_loss = compute_gan_loss(fake_pred[-1]) * self.args.lambda_gan
            G_losses['gen/g_losses'] = gan_loss
            loss += gan_loss

            
            # feat loss
            if self.args.featLoss:
                real_pred = self.netD(gt*flag)
                
                GAN_Feat_loss = torch.FloatTensor(1).fill_(0).to(self.device)
                
                # last output is the final prediction, so we exclude it
                num_intermediate_outputs = len(fake_pred) - 1
                for j in range(num_intermediate_outputs):  # for each layer output
                    unweighted_loss = self.L1Loss(
                        fake_pred[j]*flag, real_pred[j].detach()*flag) *\
                        batch_size / (flag.sum() + 1e-9)
                    GAN_Feat_loss += unweighted_loss * self.args.lambda_feat
                G_losses['gen/GAN_Feat'] = GAN_Feat_loss

            # per loss
            if self.args.per_loss_final:
                genpre_loss = self.perFinalLoss(xg*flag,gt*flag) *\
                     batch_size / (flag.sum() + 1e-9)  * self.args.lambda_gper
                G_losses['gen/per_loss'] = genpre_loss
                loss += genpre_loss

            # rec loss
            if self.args.rec_loss_final:
                genrec_loss = self.L1Loss(xg*flag,gt*flag)*\
                     batch_size / (flag.sum() + 1e-9)   * self.args.lambda_grec
                G_losses['gen/rec_loss'] = genrec_loss
                loss += genrec_loss
        
            
            if self.args.id_loss:
                id_loss = self.idLoss(self.process_id_input(xg,bbox_inp[:,:4]),
                                     self.process_id_input(torch.cat([xs,xt],0),
                                                torch.cat([xs_bbox,xt_bbox],0)[:,:4])
                                    )  * self.args.lambda_id 
                
                G_losses['gen/id_loss'] = id_loss
                
                loss += id_loss

            if self.args.params_loss:
            
                params_loss = self.paramsLoss(
                            self.process_id_input(
                                    F.interpolate(xg,[256, 256], mode='bilinear'),
                                        bbox_inp[:,-4:],224),
                            self.process_id_input(xt_inp,bbox_inp[:,-4:],224)

                )  * self.args.lambda_params_gen
                
                G_losses['gen/params_loss'] = params_loss
                loss += params_loss
            G_losses['loss'] = loss 
            return G_losses,loss,xg

        G_losses['loss'] = loss 
        return G_losses,loss,xw

    @staticmethod
    def process_id_input(x,bbox=None,size=112):

        if bbox is None:
            return F.interpolate(
                            x,
                            [size, size], mode='bilinear')
        batch = x.shape[0]
        crop_x = []
        for i in range(batch):
            x_min,y_min,x_max,y_max = bbox[i]
           
            crop_x.append(F.interpolate(
                            x[i,:,int(y_min):int(y_max),int(x_min):int(x_max)].unsqueeze(0),
                            [size, size], mode='bilinear'))
        crop_x = torch.cat(crop_x,0)
        return  crop_x
    def get_latest_losses(self):
        if self.netD is None:
            return self.g_losses
        return {**self.g_losses,**self.d_losses}

    def get_latest_generated(self):
        return self.generator

    def loadParameters(self,path):
        ckpt = torch.load(path, map_location=lambda storage, loc: storage)
        self.netG.load_state_dict(ckpt['G'],strict=False)
        self.optimG.load_state_dict(ckpt['g_optim'])
        if self.netD is not None and 'D' in ckpt:
            self.netD.load_state_dict(ckpt['D'],strict=False)
            self.optimD.load_state_dict(ckpt['d_optim'])

    def saveParameters(self,path):

        save_dict = {
            "G": self.netG_module.state_dict(),
            'g_ema':self.g_ema.state_dict(),
            "g_optim": self.optimG.state_dict(),
            "args": self.args
        }
        if self.netD is not None:
            save_dict['D'] = self.netD_module.state_dict()
            save_dict['d_optim'] = self.optimD.state_dict()
        torch.save(
                   save_dict,
                   path
                )

    def get_lr(self):
        return self.optimG.state_dict()['param_groups'][0]['lr']

    
    def select_img(self, data,size=None, name='fake', axis=2):
        if size is None:
            size = self.args.size
        data = [F.adaptive_avg_pool2d(x,size) for x in data]
        return super().select_img(data, name, axis)

    def load_scratch_path(self):
        
        state_dict = torch.load(self.args.scratch_path)['net_G_ema']
        model_dict = self.netG.state_dict()
        pretrained_dict = {k:v for k,v in state_dict.items() if k in model_dict} 
        model_dict.update(pretrained_dict)
        self.netG.load_state_dict(model_dict)
        print('load scratch model')
    

    def get_loss_from_val(self,loss):
        
        if self.args.stage == 'warp':
            return super().get_loss_from_val(loss)
        else:
            return loss['gen/per_loss'] + loss['gen/rec_loss'] + loss['gen/id_loss'] + loss['gen/params_loss']

    
    def freeze_models(self):
       
        for n in self.args.frozen_params:
            for p in self.netG.__getattr__(n).parameters():
                p.requires_grad = False 


        


    
    

    
