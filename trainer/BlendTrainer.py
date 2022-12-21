#! /usr/bin/python 
# -*- encoding: utf-8 -*-
'''
@author LeslieZhao
@date 20221221
'''
import torch 

from trainer.ModelTrainer import ModelTrainer
from model.BlendModule.generator import Generator
from model.AlignModule.discriminator import Discriminator
from model.AlignModule.criterion import *
from utils.utils import *
import torch.nn.functional as F
import random
import torch.distributed as dist

class BlendTrainer(ModelTrainer):

    def __init__(self, args):
        super().__init__(args)
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = 'cuda'
      
        self.netG = Generator(args).to(self.device)
        self.g_ema = Generator(args).to(self.device)
        self.g_ema.eval()

        self.netD = Discriminator(args.d_input_nc).to(self.device)

        init_weights(self.netD,'xavier_uniform')
        init_weights(self.netG,'xavier_uniform')

        self.optimG,self.optimD = self.create_optimizer() 

        if args.pretrain_path is not None:
            self.loadParameters(args.pretrain_path)

        accumulate(self.g_ema,self.netG,0)
        if args.dist:
            self.netG,self.netG_module = self.use_ddp(self.netG)
            self.netD,self.netD_module = self.use_ddp(self.netD)
        else:
            self.netG_module = self.netG 
            self.netD_module = self.netD
        
        if self.args.per_loss:
            self.perLoss = PerceptualLoss(
                                use_style_loss=False,
                                num_scales=args.per_num_scales).to(self.device)
            self.perLoss.eval()
        
        if self.args.rec_loss:
            self.L1Loss = torch.nn.L1Loss()

        self.accum = 0.5 ** (32 / (10 * 1000))
        

    def create_optimizer(self):
        g_optim = torch.optim.Adam(
                    self.netG.parameters(),
                    lr=self.args.g_lr,
                    betas=(self.args.beta1, self.args.beta2),
                    )
        d_optim = torch.optim.Adam(
                    self.netD.parameters(),
                    lr=self.args.d_lr,
                    betas=(self.args.beta1, self.args.beta2),
                    )
        
        return  g_optim,d_optim

    
    def run_single_step(self, data, steps):
        self.netG.train() 
        super().run_single_step(data, steps)
        

    def run_discriminator_one_step(self, data,step):
        
        D_losses = {}
        requires_grad(self.netG, False)
        requires_grad(self.netD, True)

        I_a,I_gray,I_t,hat_t,M_a,M_t,M_hat,gt = data 
        fake,M_Ah,M_Ai,_ = self.netG(I_a,I_gray,I_t,M_a,M_t,gt,train=True)
        fake_pred = self.netD(torch.cat([fake,M_Ah,M_Ai],1))
        real_pred = self.netD(torch.cat([gt,M_Ah,M_Ai],1))
        d_loss = compute_dis_loss(fake_pred[-1],real_pred[-1],D_losses)

        D_losses['d'] = d_loss
        
        self.optimD.zero_grad()
        d_loss.backward()
        self.optimD.step()
        
        self.d_losses = D_losses


    def run_generator_one_step(self, data,step):
        
        
        requires_grad(self.netG, True)
        requires_grad(self.netD, False)
        
        I_a,I_gray,I_t,hat_t,M_a,M_t,M_hat,gt = data 
        G_losses,loss,xg,warp_img = self.compute_g_loss(I_a,I_gray,I_t,M_a,M_t,gt)
        self.optimG.zero_grad()
        loss.mean().backward()
        self.optimG.step()

        g_losses,loss,fake_nopair,label_nopair,no_pair_warp = self.compute_cycle_g_loss(I_a,I_gray,I_t,hat_t,M_a,M_t,M_hat)
        self.optimG.zero_grad()
        loss.mean().backward()
        self.optimG.step()
        
        accumulate(self.g_ema,self.netG_module,self.accum)

        self.g_losses = {**G_losses,**g_losses}
        
        self.generator = [I_a.detach(),warp_img.detach(),
                        no_pair_warp.detach(),
                        fake_nopair.detach(),
                        label_nopair.detach(),xg.detach(),gt.detach()]
        
    
    def evalution(self,test_loader,steps,epoch):
        
        loss_dict = {}
        index = random.randint(0,len(test_loader)-1)
        counter = 0
        with torch.no_grad():
            for i,data in enumerate(test_loader):
                
                data = self.process_input(data)
                I_a,I_gray,I_t,hat_t,M_a,M_t,M_hat,gt = data 
                G_losses,losses,xg,warp_img = self.compute_g_loss(I_a,I_gray,I_t,M_a,M_t,gt)
                for k,v in G_losses.items():
                    loss_dict[k] = loss_dict.get(k,0) + v.detach()
                if i == index and self.args.rank == 0 :
                    fake = self.g_ema(I_a,I_gray,I_t,M_a,M_t,gt,train=False)
                    show_data = [I_a,warp_img,xg,fake,gt]
                    self.val_vis.display_current_results(self.select_img(show_data),steps)
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
       

    def compute_g_loss(self,I_a,I_gray,I_t,M_a,M_t,gt):
        G_losses = {}
        loss = 0
       
        fake,M_Ah,M_Ai,warp_img = self.netG(I_a,I_gray,I_t,M_a,M_t,gt,train=True)
        fake_pred = self.netD(torch.cat([fake,M_Ah,M_Ai],1))
        gan_loss = compute_gan_loss(fake_pred[-1]) * self.args.lambda_gan
        G_losses['g_losses'] = gan_loss
        loss += gan_loss

        # feat loss
        if self.args.featLoss:
            GAN_Feat_loss = torch.FloatTensor(1).fill_(0).to(self.device)
            real_pred = self.netD(torch.cat([gt,M_Ah,M_Ai],1))
            
            # last output is the final prediction, so we exclude it
            num_intermediate_outputs = len(fake_pred) - 1
            for j in range(num_intermediate_outputs):  # for each layer output
                unweighted_loss = self.L1Loss(
                    fake_pred[j], real_pred[j].detach())
                GAN_Feat_loss += unweighted_loss * self.args.lambda_feat 
            G_losses['GAN_Feat'] = GAN_Feat_loss
    
        if self.args.rec_loss:
            rec_loss = self.L1Loss(fake,gt) * self.args.lambda_rec 
            G_losses['rec_loss'] = rec_loss
            loss += rec_loss
        

        if self.args.per_loss:
            per_loss = self.perLoss(fake,gt).mean() * self.args.lambda_per 
            G_losses['per_loss'] = per_loss
            loss += per_loss 
        
        G_losses['loss'] = loss

        return G_losses,loss,fake,warp_img

    
    def compute_cycle_g_loss(self,I_a,I_gray,I_t,hat_t,M_a,M_t,M_hat):
        G_losses = {}
        loss = 0
        fake_pair_h,fake_pair_i,label_pair_h,label_pair_i,_ = self.netG(I_a,I_gray,I_t,M_a,M_t,cycle=True)
        fake_nopair_h,fake_nopair_i,label_nopair_h,label_nopair_i,no_pair_warp = self.netG(I_a,I_gray,hat_t,M_a,M_hat,cycle=True)
       
        loss = self.L1Loss(fake_pair_h,label_pair_h) + \
             self.L1Loss(fake_pair_i,label_pair_i) + \
            self.L1Loss(fake_nopair_h,label_nopair_h) + \
            self.L1Loss(fake_nopair_i,label_nopair_i)
        loss = loss * self.args.lambda_cycle
        G_losses['cycle'] = loss
       
        
        fake_nopair = F.interpolate(fake_nopair_h+fake_nopair_i, size=I_t.shape[-2:],mode='bilinear')
        label_nopair = F.interpolate(label_nopair_h+label_nopair_i, size=I_t.shape[-2:],mode='bilinear')
        return G_losses,loss,fake_nopair,label_nopair,no_pair_warp

    
    def get_latest_losses(self):
        return {**self.g_losses,**self.d_losses}

    def get_latest_generated(self):
        return self.generator

    def loadParameters(self,path):
        ckpt = torch.load(path, map_location=lambda storage, loc: storage)
        self.netG.load_state_dict(ckpt['G'],strict=False)
        self.netD.load_state_dict(ckpt['D'],strict=False)
        self.optimG.load_state_dict(ckpt['g_optim'])
        self.optimD.load_state_dict(ckpt['d_optim'])

    def saveParameters(self,path):
        torch.save(
                    {
                        "G": self.netG_module.state_dict(),
                        'g_ema':self.g_ema.state_dict(),
                        "D": self.netD_module.state_dict(),
                        "g_optim": self.optimG.state_dict(),
                        "d_optim": self.optimD.state_dict(),
                        "args": self.args,
                    },
                    path
                )

    def get_lr(self):
        return self.optimG.state_dict()['param_groups'][0]['lr']

    def get_loss_from_val(self,loss):
        return loss['rec_loss'] + loss['per_loss']
    


    
            


        


    
    

    
