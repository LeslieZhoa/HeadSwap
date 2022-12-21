from torch import nn
from torch.nn.utils import spectral_norm

# PyTorch implementation by vinesmsuic
# Referenced from official tensorflow implementation: https://github.com/SystemErrorWang/White-box-Cartoonization/blob/master/train_code/network.py
# slim.convolution2d uses constant padding (zeros).
# Paper used spectral_norm

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,activate=True):
        super().__init__()
        self.sn_conv = spectral_norm(nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride, 
                padding,
                padding_mode="zeros" # Author's code used slim.convolution2d, which is using SAME padding (zero padding in pytorch) 
            ))
        self.activate = activate
        if self.activate:
            self.LReLU = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x = self.sn_conv(x)
        if self.activate:
            x = self.LReLU(x)

        return x


class Discriminator(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[32, 64, 128,256]):
        super().__init__()
        
        self.model = nn.Sequential(
            #k3n32s2
            Block(in_channels, features[0], kernel_size=3, stride=2, padding=1),
            #k3n32s1
            Block(features[0], features[0], kernel_size=3, stride=1, padding=1),

            #k3n64s2
            Block(features[0], features[1], kernel_size=3, stride=2, padding=1),
            #k3n64s1
            Block(features[1], features[1], kernel_size=3, stride=1, padding=1),

            #k3n128s2
            Block(features[1], features[2], kernel_size=3, stride=2, padding=1),
            #k3n128s1
            Block(features[2], features[2], kernel_size=3, stride=1, padding=1),

            #k3n256s2
            Block(features[2], features[3], kernel_size=3, stride=2, padding=1),
            #k3n256s1
            Block(features[3], features[3], kernel_size=3, stride=1, padding=1),


            #k1n1s1
            Block(features[3], out_channels, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x):
        y = []
        for m in self.model:
            x = m(x)
            y.append(x)

        return y
    