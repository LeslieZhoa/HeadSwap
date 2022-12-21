
import torch
import torch.nn.functional as F
def compute_dis_loss(fake_pred,real_pred,D_loss):
    # d_real = torch.relu(1. - real_pred).mean() 
    # d_fake = torch.relu(1. + fake_pred).mean() 
    d_real = F.mse_loss(real_pred,torch.ones_like(real_pred))
    d_fake = F.mse_loss(fake_pred, torch.zeros_like(fake_pred))

    D_loss['d_real'] = d_real 
    D_loss['d_fake'] = d_fake 
    return d_real + d_fake 

def compute_gan_loss(fake_pred):

    return F.mse_loss(fake_pred,torch.ones_like(fake_pred))