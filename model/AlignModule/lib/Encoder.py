from torch import nn
import torchvision

class EncoderNet(nn.Module):
    def __init__(self,encoder_num):
        super().__init__()
        self.encoder = torchvision.models.mobilenet_v2(num_classes=encoder_num)

    def forward(self,x):
       
        return self.encoder(x)