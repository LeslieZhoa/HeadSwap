from model.third.Deep3dRec.network import ReconNetWrapper
import torch.nn as nn
import torch
class ParamsLoss(nn.Module):
    def __init__(self,pretrain_model, requires_grad=False):
        super(ParamsLoss, self).__init__()
        self.model = ReconNetWrapper()
        self.model.load_state_dict(torch.load(pretrain_model)['net_recon'])
        self.model.eval()

        self.index = list(range(80,144)) + \
                    list(range(224,227)) + \
                    list(range(254,257))
        self.L1Loss = nn.L1Loss()

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

        

    def forward(self, x1, x2):
        y1 = self.model(self.process(x1))
        y2 = self.model(self.process(x2))
        
        loss = self.L1Loss(y1[:,self.index],y2[:,self.index])
        return loss.mean()

    def process(self,x):
        return (x + 1.0) * 0.5