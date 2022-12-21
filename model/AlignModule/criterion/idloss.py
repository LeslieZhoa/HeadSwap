from torch.nn import functional as F
from torch import nn
import torch
from .arcface import Backbone


class IDLoss(nn.Module):
    def __init__(self,pretrain_model, requires_grad=False):
        super(IDLoss, self).__init__()
        self.idModel = Backbone(50,0.6,'ir_se')
        self.idModel.load_state_dict(torch.load(pretrain_model),strict=False)
        self.idModel.eval()
        self.criterion = nn.CosineSimilarity(dim=1,eps=1e-6)
        self.id_size = 112
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

        

    def forward(self, x, y):
        x_id, _ = self.idModel(x)
        y_id,_ = self.idModel(y)
        loss = 1 - self.criterion(x_id,y_id)
        return loss.mean()