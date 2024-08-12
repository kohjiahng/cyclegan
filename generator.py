from blocks import ResBlock
from configparser import ConfigParser
from blocks import ConvInstanceNormRelu, ConvTransposeInstanceNormRelu
from torch import nn
import torch
config = ConfigParser()
config.read('config.ini')

IMG_RES = config.getint('params','IMG_RES')

class ResNetGenerator(nn.Module):
    def __init__(self, n_resblocks):
        super().__init__()

        self.enc = nn.Sequential(
            ConvInstanceNormRelu(64,kernel_size=(7,7),stride=1,padding="same"),
            ConvInstanceNormRelu(128,kernel_size=(3,3),stride=2),
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

class UNetGenerator(nn.Module):
    def __init__(self):
        super().__init__()

        enc_kwargs = {
            'kernel_size': (4,4),
            'stride': 2,
            'padding': 1,
        }

        self.enc = nn.Sequential(
            ConvInstanceNormRelu(64,norm=False,**enc_kwargs),
            ConvInstanceNormRelu(128,**enc_kwargs),
            ConvInstanceNormRelu(256,**enc_kwargs),
            ConvInstanceNormRelu(512,**enc_kwargs),
            ConvInstanceNormRelu(512,**enc_kwargs),
            ConvInstanceNormRelu(512,**enc_kwargs),
            ConvInstanceNormRelu(512,**enc_kwargs),
            ConvInstanceNormRelu(512,norm=False,**enc_kwargs),
        )
        dec_kwargs = {
            'kernel_size': (4,4),
            'stride': 2,
            'padding': 1
        }
        self.dec = nn.Sequential(
            ConvTransposeInstanceNormRelu(512,**dec_kwargs,dropout=True),
            ConvTransposeInstanceNormRelu(512,**dec_kwargs,dropout=True),
            ConvTransposeInstanceNormRelu(512,**dec_kwargs,dropout=True),
            ConvTransposeInstanceNormRelu(512,**dec_kwargs),
            ConvTransposeInstanceNormRelu(256,**dec_kwargs),
            ConvTransposeInstanceNormRelu(128,**dec_kwargs),
            ConvTransposeInstanceNormRelu(64,**dec_kwargs),
        )
        self.feature_block = nn.Sequential(
            nn.ConvTranspose2d(128, 3, kernel_size=(4,4), stride=2,padding=1),
            nn.Tanh()
        )
    def forward(self, x):
        skips = []
        for down in self.enc:
            x = down(x)
            skips.append(x)
        skips = reversed(skips[:-1])
        for up, skip in zip(self.dec, skips):
            x = up(x)
            x = torch.cat([x, skip], dim = 1)
        return self.feature_block(x)

    
def get_gen(gen_name, hid_channels, num_resblocks, in_channels=3, out_channels=3):
    if gen_name == "unet":
        return UNetGenerator(hid_channels, in_channels, out_channels)
    elif gen_name == "resnet":
        return ResNetGenerator(hid_channels, in_channels, out_channels, num_resblocks)
    else:
        raise NotImplementedError(f"Generator name '{gen_name}' not recognized.")
if __name__ == '__main__':
    gen = ResNetGenerator(6)
    inp = torch.zeros((1, 3, IMG_RES, IMG_RES))
    out = gen(inp)
    print(f"ResNet output shape: {out.shape}")

    gen = UNetGenerator()
    inp = torch.zeros((1, 3, IMG_RES, IMG_RES))
    out = gen(inp)
    print(f"UNet output shape: {out.shape}")