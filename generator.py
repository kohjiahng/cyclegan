from blocks import ResBlock
from configparser import ConfigParser
from blocks import ConvInstanceNormRelu, ConvTransposeInstanceNormRelu
from torch import nn
import torch
config = ConfigParser()
config.read('config.ini')

IMG_RES = config.getint('params','IMG_RES')

class Generator(nn.Module):
    def __init__(self, n_resblocks):
        super().__init__()

        self.enc = nn.Sequential(
            ConvInstanceNormRelu(3,kernel_size=(7,7),stride=1,padding="same"),
            ConvInstanceNormRelu(64,kernel_size=(3,3),stride=2),
            ConvInstanceNormRelu(256,kernel_size=(3,3),stride=2),
        )
        self.resblocks = nn.Sequential(
            *(ResBlock(256,3) for _ in range(n_resblocks))
        )
        self.dec = nn.Sequential(
            ConvTransposeInstanceNormRelu(128,kernel_size=(3,3),stride=2),
            ConvTransposeInstanceNormRelu(64,kernel_size=(3,3),stride=2,output_padding=1),
            nn.Conv2d(64, 3, kernel_size=(7,7), stride=1,padding="same"),
            nn.Tanh()
        )
        self.model = nn.Sequential(
            self.enc,
            self.resblocks,
            self.dec
        )

    def forward(self, X):
        return self.model(X)

if __name__ == '__main__':
    gen = Generator(6)
    inp = torch.zeros((1, 3, IMG_RES, IMG_RES))
    out = gen(inp)
    print(f"Output shape: {out.shape}")