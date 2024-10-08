from utils import ImagePool
from generator import ResNetGenerator, UNetGenerator
from discriminator import Discriminator
from configparser import ConfigParser
import torch
import itertools
from torch import nn

config = ConfigParser()
config.read('config.ini')

POOL_SIZE = config.getint('params', 'POOL_SIZE')
LAMBDA = config.getint('params', 'LAMBDA')
IMG_RES = config.getint('params', 'IMG_RES')

class CycleGAN():
    def __init__(self, gan_loss_fn = 'bce', generator = 'unet', n_resblocks=6):
        '''
        gan_loss_fn: either 'mse' or 'bce'
        '''
        if generator == 'unet':
            self.genF = UNetGenerator()
            self.genG = UNetGenerator()
        else:
            self.genF = ResNetGenerator(n_resblocks)
            self.genG = ResNetGenerator(n_resblocks)

        self.discA = Discriminator()
        self.discB = Discriminator()

        self.pool_A = ImagePool(POOL_SIZE)
        self.pool_B = ImagePool(POOL_SIZE)

        if gan_loss_fn == 'mse':
            self.gan_loss_fn = nn.MSELoss()
        elif gan_loss_fn == 'bce':
            self.gan_loss_fn = nn.BCELoss()
        else:
            raise NotImplementedError
        
        self.cycle_loss_fn = nn.L1Loss()
        self.identity_loss_fn = nn.L1Loss()

    def disc_loss(self, real_A, fake_A, real_B, fake_B):
        fake_A = self.pool_A.query(fake_A)
        fake_B = self.pool_B.query(fake_B)

        real_A_score = self.discA(real_A)
        fake_A_score = self.discA(fake_A)
        loss_A = self.gan_loss_fn(
            torch.concat((real_A_score,fake_A_score)),
            torch.concat((torch.ones_like(real_A_score), torch.zeros_like(fake_A_score)))
        )

        real_B_score = self.discB(real_B)
        fake_B_score = self.discB(fake_B)
        loss_B = self.gan_loss_fn(
            torch.concat((real_B_score,fake_B_score)),
            torch.concat((torch.ones_like(real_B_score), torch.zeros_like(fake_B_score)))
        )

        return loss_A + loss_B
    
    def gan_loss(self, fake_A, fake_B):
        fake_A_score = self.discA(fake_A)
        loss_A = self.gan_loss_fn(
            fake_A_score,
            torch.ones_like(fake_A_score)
        )

        fake_B_score = self.discB(fake_B)
        loss_B = self.gan_loss_fn(
            fake_B_score,
            torch.ones_like(fake_B_score)
        )

        return loss_A + loss_B
 
    def cycle_loss(self, real_A, fake_A, real_B, fake_B):
        regen_A = self.genG(fake_B)
        regen_B = self.genF(fake_A)

        return self.cycle_loss_fn(real_A, regen_A) + self.cycle_loss_fn(real_B, regen_B)

    def identity_loss(self, real_A, real_B):
        idt_A = self.genG(real_A)
        idt_B = self.genF(real_B)

        return self.identity_loss_fn(real_A, idt_A) + self.identity_loss_fn(real_B, idt_B)

    # ---------------------------------- HELPERS --------------------------------- #
    def infer_B(self, X): # A to B
        return self.genF(X)
    
    def infer_A(self, X): # B to A
        return self.genG(X)
    
    def get_disc_parameters(self):
        return itertools.chain(self.discA.parameters(), self.discB.parameters())
    
    def get_gen_parameters(self):
        return itertools.chain(self.genF.parameters(), self.genG.parameters())

    def eval(self):
     for module in self.modules():
            module.eval()
    def train(self):
        for module in self.modules():
            module.train()
   
    def cuda(self):
        for module in self.modules():
            module.to('cuda')
        return self
    def apply(self,fn):
        for module in self.modules():
            module.apply(fn)

    def ckpt(self):
        return {
            'genF_state_dict': self.genF.state_dict(),
            'genG_state_dict': self.genG.state_dict(),
            'discA_state_dict': self.discA.state_dict(),
            'discB_state_dict': self.discB.state_dict(),
        }
    def modules(self):
        return self.genF, self.genG, self.discA, self.discB
    def init_params(self):
        for module in self.modules():
            module(torch.zeros((1,3,IMG_RES,IMG_RES), device='cuda'))
