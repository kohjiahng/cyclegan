import torch
from torch import nn
from configparser import ConfigParser
from blocks import ConvInstanceNormRelu 

config = ConfigParser()
config.read('config.ini')

IMG_RES = config.getint('params','IMG_RES')
BATCH_SIZE = config.getint('params','BATCH_SIZE')
DISC_NOISE_STD = config.getfloat('params', 'DISC_NOISE_STD')

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(
            ConvInstanceNormRelu(64,kernel_size=(4,4),stride=2,padding=1,leaky=True,norm=False),
            ConvInstanceNormRelu(128,kernel_size=(4,4),stride=2,padding=1,leaky=True),
            ConvInstanceNormRelu(256,kernel_size=(4,4),stride=2,padding=1,leaky=True),
            ConvInstanceNormRelu(512,kernel_size=(4,4),stride=1,padding=1,leaky=True),
            nn.Conv2d(512,1,kernel_size=(4,4), stride=1, padding=1),
        )

    def forward(self, X):
        return self.model(X)

if __name__ == '__main__':
    disc = Discriminator()
    inp = torch.zeros((1, 3, IMG_RES, IMG_RES))
    out = disc(inp)
    print(f"Output shape: {out.shape}")
