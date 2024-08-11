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
    def __init__(self, filters, leaky=False, norm=True,**kwargs):
        super().__init__()
        self.block = nn.Sequential(
            nn.LazyConv2d(filters, bias=False,**kwargs),
            *([nn.LazyInstanceNorm2d(filters, affine=True)] if norm else []),
            nn.LeakyReLU(0.2) if leaky else nn.ReLU()
        )
    def forward(self, X):
        return self.block(X)

class ConvTransposeInstanceNormRelu(nn.Module):
    def __init__(self, filters, dropout=False, **kwargs): 
        super().__init__()
        self.block = nn.Sequential(
            nn.LazyConvTranspose2d(filters,bias=False, **kwargs),
            nn.LazyInstanceNorm2d(filters, affine=True),
            *([nn.Dropout(0.5)] if dropout else []),
            nn.ReLU()
        )
    def forward(self, X):
        return self.block(X)


class Upsampling(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=4,
        stride=2,
        padding=1,
        output_padding=0,
        dropout=False,
    ):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels,
                               kernel_size=kernel_size, stride=stride, 
                               padding=padding, output_padding=output_padding, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True),
        )
        if dropout:
            self.block.append(nn.Dropout(0.5))
        self.block.append(nn.ReLU(True))
        
    def forward(self, x):
        return self.block(x)
    
class Downsampling(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=4,
        stride=2,
        padding=1,
        norm=True,
        lrelu=True,
    ):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=kernel_size, stride=stride, padding=padding, bias=not norm),
        )
        if norm:
            self.block.append(nn.InstanceNorm2d(out_channels, affine=True))
        if lrelu is not None:
            self.block.append(nn.LeakyReLU(0.2, True) if lrelu else nn.ReLU(True))
        
    def forward(self, x):
        return self.block(x)