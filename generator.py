from blocks import ResBlock
from configparser import ConfigParser
from blocks import ConvInstanceNormRelu, ConvTransposeInstanceNormRelu, Downsampling, Upsampling
from torch import nn
import torch
config = ConfigParser()
config.read('config.ini')

IMG_RES = config.getint('params','IMG_RES')

# class ResNetGenerator(nn.Module):
#     def __init__(self, n_resblocks):
#         super().__init__()

#         self.enc = nn.Sequential(
#             ConvInstanceNormRelu(64,kernel_size=(7,7),stride=1,padding="same"),
#             ConvInstanceNormRelu(128,kernel_size=(3,3),stride=2),
#             ConvInstanceNormRelu(256,kernel_size=(3,3),stride=2),
#         )
#         self.resblocks = nn.Sequential(
#             *(ResBlock(256,3) for _ in range(n_resblocks))
#         )
#         self.dec = nn.Sequential(
#             ConvTransposeInstanceNormRelu(128,kernel_size=(3,3),stride=2),
#             ConvTransposeInstanceNormRelu(64,kernel_size=(3,3),stride=2,output_padding=1),
#             nn.Conv2d(64, 3, kernel_size=(7,7), stride=1,padding="same"),
#             nn.Tanh()
#         )
#         self.model = nn.Sequential(
#             self.enc,
#             self.resblocks,
#             self.dec
#         )

#     def forward(self, X):
#         return self.model(X)

# class UNetGenerator(nn.Module):
#     def __init__(self):
#         super().__init__()

#         enc_kwargs = {
#             'kernel_size': (4,4),
#             'stride': 2,
#             'padding': 1
#         }

#         self.enc = nn.Sequential(
#             ConvInstanceNormRelu(64,**enc_kwargs),
#             ConvInstanceNormRelu(128,**enc_kwargs),
#             ConvInstanceNormRelu(256,**enc_kwargs),
#             ConvInstanceNormRelu(512,**enc_kwargs),
#             ConvInstanceNormRelu(512,**enc_kwargs),
#             ConvInstanceNormRelu(512,**enc_kwargs),
#             ConvInstanceNormRelu(512,**enc_kwargs),
#             ConvInstanceNormRelu(512,norm=False,**enc_kwargs),
#         )
#         dec_kwargs = {
#             'kernel_size': (4,4),
#             'stride': 2,
#             'padding': 1
#         }
#         self.dec = nn.Sequential(
#             ConvTransposeInstanceNormRelu(512,**dec_kwargs,dropout=True),
#             ConvTransposeInstanceNormRelu(512,**dec_kwargs,dropout=True),
#             ConvTransposeInstanceNormRelu(512,**dec_kwargs,dropout=True),
#             ConvTransposeInstanceNormRelu(512,**dec_kwargs),
#             ConvTransposeInstanceNormRelu(256,**dec_kwargs),
#             ConvTransposeInstanceNormRelu(128,**dec_kwargs),
#             ConvTransposeInstanceNormRelu(64,**dec_kwargs),
#         )
#         self.feature_block = nn.Sequential(
#             nn.ConvTranspose2d(128, 3, kernel_size=(4,4), stride=2,padding=1),
#             nn.Tanh()
#         )
#     def forward(self, x):
#         skips = []
#         for down in self.enc:
#             x = down(x)
#             skips.append(x)
#         skips = reversed(skips[:-1])
#         for up, skip in zip(self.dec, skips):
#             x = up(x)
#             x = torch.cat([x, skip], dim = 1)
#         return self.feature_block(x)

class UNetGenerator(nn.Module):
    def __init__(self, hid_channels=64, in_channels=3, out_channels=3):
        super().__init__()
        self.downsampling_path = nn.Sequential(
            Downsampling(in_channels, hid_channels, norm=False), # 64x128x128 out_channels-height-width
            Downsampling(hid_channels, hid_channels*2), # 128x64x64
            Downsampling(hid_channels*2, hid_channels*4), # 256x32x32
            Downsampling(hid_channels*4, hid_channels*8), # 512x16x16
            Downsampling(hid_channels*8, hid_channels*8), # 512x8x8
            Downsampling(hid_channels*8, hid_channels*8), # 512x4x4
            Downsampling(hid_channels*8, hid_channels*8), # 512x2x2
            Downsampling(hid_channels*8, hid_channels*8, norm=False), # 512x1x1, instance norm does not work on 1x1
        )
        self.upsampling_path = nn.Sequential(
            Upsampling(hid_channels*8, hid_channels*8, dropout=True), # (512+512)x2x2
            Upsampling(hid_channels*16, hid_channels*8, dropout=True), # (512+512)x4x4
            Upsampling(hid_channels*16, hid_channels*8, dropout=True), # (512+512)x8x8
            Upsampling(hid_channels*16, hid_channels*8), # (512+512)x16x16
            Upsampling(hid_channels*16, hid_channels*4), # (256+256)x32x32
            Upsampling(hid_channels*8, hid_channels*2), # (128+128)x64x64
            Upsampling(hid_channels*4, hid_channels), # (64+64)x128x128
        )
        self.feature_block = nn.Sequential(
            nn.ConvTranspose2d(hid_channels*2, out_channels,
                               kernel_size=4, stride=2, padding=1), # 3x256x256
            nn.Tanh(), # hyperbolic tangent
        )
        
    def forward(self, x):
        # Downsampling through the model
        skips = []
        for down in self.downsampling_path:
            x = down(x)
            skips.append(x)
        skips = reversed(skips[:-1])
        
        # Upsampling and establishing the skip connections
        for up, skip in zip(self.upsampling_path, skips):
            x = up(x)
            x = torch.cat([x, skip], dim=1)
        return self.feature_block(x)
    
class ResNetGenerator(nn.Module):
    def __init__(self,n_resblocks, hid_channels=64, in_channels=3, out_channels=3):
        super().__init__()
        self.model = nn.Sequential(
            # downsampling path
            nn.ReflectionPad2d(3), # to handle border pixels
            Downsampling(in_channels, hid_channels,
                         kernel_size=7, stride=1, padding=0, lrelu=False), # 64x256x256
            Downsampling(hid_channels, hid_channels*2, kernel_size=3, lrelu=False), # 128x128x128
            Downsampling(hid_channels*2, hid_channels*4, kernel_size=3, lrelu=False), # 256x64x64
            
            # residual blocks
            *[ResBlock(hid_channels*4) for _ in range(n_resblocks)], # 256x64x64
            
            # upsampling path
            Upsampling(hid_channels*4, hid_channels*2, kernel_size=3, output_padding=1), # 128x128x128
            Upsampling(hid_channels*2, hid_channels, kernel_size=3, output_padding=1), # 64x256x256
            nn.ReflectionPad2d(3), # to handle border pixels
            nn.Conv2d(hid_channels, out_channels, kernel_size=7, stride=1, padding=0), # 3x256x256
            nn.Tanh(),# pixels in the range [-1,1]
        )
        
    def forward(self, x):
        return self.model(x)
    
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