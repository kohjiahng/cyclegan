import torch
from torch import nn
from configparser import ConfigParser
from blocks import ConvInstanceNormRelu, Downsampling

config = ConfigParser()
config.read('config.ini')

IMG_RES = config.getint('params','IMG_RES')
BATCH_SIZE = config.getint('params','BATCH_SIZE')
DISC_NOISE_STD = config.getfloat('params', 'DISC_NOISE_STD')

# class Discriminator(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.model = torch.nn.Sequential(
#             ConvInstanceNormRelu(64,kernel_size=(4,4),stride=2,padding=1,leaky=True,norm=False),
#             ConvInstanceNormRelu(128,kernel_size=(4,4),stride=2,padding=1,leaky=True),
#             ConvInstanceNormRelu(256,kernel_size=(4,4),stride=2,padding=1,leaky=True),
#             ConvInstanceNormRelu(512,kernel_size=(4,4),stride=1,padding=1,leaky=True),
#             nn.Conv2d(512,1,kernel_size=(4,4), stride=1, padding=1),
#         )

#     def forward(self, X):
#         return self.model(X)

class Discriminator(nn.Module):
    def __init__(self, hid_channels=64, in_channels=3):
        super().__init__()
        self.block = nn.Sequential(
            Downsampling(in_channels, hid_channels, norm=False), # 64x128x128
            Downsampling(hid_channels, hid_channels*2), # 128x64x64
            Downsampling(hid_channels*2, hid_channels*4), # 256x32x32
            Downsampling(hid_channels*4, hid_channels*8, stride=1), # 512x31x31
            nn.Conv2d(hid_channels*8, 1, kernel_size=4, padding=1), # 1x30x30 (num_channels-h-w)
        )# 1 channel for binary classification task, 30-30 spatial dimensions of the feature map
        
    def forward(self, x):
        return self.block(x)

if __name__ == '__main__':
    disc = Discriminator()
    inp = torch.zeros((1, 3, IMG_RES, IMG_RES))
    out = disc(inp)
    print(f"Output shape: {out.shape}")
