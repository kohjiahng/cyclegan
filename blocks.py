from torch import nn
from configparser import ConfigParser

config = ConfigParser()
config.read('config.ini')
IMG_RES = config.getint('params', 'IMG_RES')

class ResBlock(nn.Module):
    def __init__(self, filters = 256, kernel_size = 3):
        super().__init__()
        self.conv1 = nn.LazyConv2d(filters, kernel_size=kernel_size, stride=1, padding='same')
        self.conv2 = nn.LazyConv2d(filters, kernel_size=kernel_size, stride=1, padding='same')
        self.relu = nn.ReLU()
    def forward(self, X):
        x = self.relu(self.conv1(X))
        y = self.relu(self.conv2(x))
        return X+y 

class ConvInstanceNormRelu(nn.Module):
    def __init__(self, filters, leaky=False, **kwargs):
        super().__init__()
        self.block = nn.Sequential(
            nn.LazyConv2d(filters, **kwargs),
            nn.LazyInstanceNorm2d(filters),
            nn.LeakyReLU(0.2) if leaky else nn.ReLU()
        )
    def forward(self, X):
        return self.block(X)

class ConvTransposeInstanceNormRelu(nn.Module):
    def __init__(self, filters, **kwargs): 
        super().__init__()
        self.block = nn.Sequential(
            nn.LazyConvTranspose2d(filters, **kwargs),
            nn.LazyInstanceNorm2d(filters),
            nn.ReLU()
        )
    def forward(self, X):
        return self.block(X)