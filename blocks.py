from torch import nn
from configparser import ConfigParser

config = ConfigParser()
config.read('config.ini')
IMG_RES = config.getint('params', 'IMG_RES')

class ResBlock(nn.Module):
    def __init__(self, filters = 256, kernel_size = 3,padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(padding),
            ConvInstanceNormRelu(filters, kernel_size=kernel_size, stride=1, padding=0),
            nn.ReflectionPad2d(padding),
            ConvInstanceNormRelu(filters, kernel_size=kernel_size, stride=1, padding=0),
        )
    def forward(self, x):
        return x + self.block(x)

class ConvInstanceNormRelu(nn.Module):
    def __init__(self, filters, leaky=False, **kwargs):
        super().__init__()
        self.block = nn.Sequential(
            nn.LazyConv2d(filters, bias=False,**kwargs),
            nn.LazyInstanceNorm2d(filters),
            nn.LeakyReLU(0.2) if leaky else nn.ReLU()
        )
    def forward(self, X):
        return self.block(X)

class ConvTransposeInstanceNormRelu(nn.Module):
    def __init__(self, filters, **kwargs): 
        super().__init__()
        self.block = nn.Sequential(
            nn.LazyConvTranspose2d(filters,bias=False, **kwargs),
            nn.LazyInstanceNorm2d(filters),
            nn.ReLU()
        )
    def forward(self, X):
        return self.block(X)