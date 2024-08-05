from utils import ImagePool
from generator import Generator
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
    def __init__(self, gan_loss_fn = 'bce', n_resblocks=6):
        '''
        gan_loss_fn: either 'mse' or 'bce'
        '''

        self.genF = Generator(n_resblocks)
        self.discB = Discriminator()

        self.genG = Generator(n_resblocks)
        self.discA = Discriminator()

        self.poolA = ImagePool(POOL_SIZE)
        self.poolB = ImagePool(POOL_SIZE)

        if gan_loss_fn == 'mse':
            self.gan_loss_fn = nn.MSELoss()
        elif gan_loss_fn == 'bce':
            self.gan_loss_fn = nn.BCELoss()
        else:
            raise NotImplementedError
        
        self.cycle_loss_fn = nn.L1Loss()
        self.identity_loss_fn = nn.L1Loss()

    # ---------------------------- TRAINING FUNCTIONS ---------------------------- #
    def forward_A(self, X):
        realA = X
        realAscore = self.discA(X)
        fakeB = self.genF(X)
        fakeBscore = self.discB(fakeB)
        realA_regen = self.genG(fakeB)

        return realA, realAscore, fakeB, fakeBscore, realA_regen

    def forward_B(self, X):
        realB = X
        realBscore = self.discB(X)
        fakeA = self.genG(X)
        fakeAscore = self.discA(fakeA)
        realB_regen = self.genF(fakeA)

        return realB, realBscore, fakeA, fakeAscore, realB_regen 

    # ---------------------------------- LOSSES ---------------------------------- #
    def disc_loss_A(self, realAscore, fakeA):
        # Same as gan_loss_A, except targets are swapped and fakeA is randomly replaced from a buffer and targets are swapped
        # Discriminator aims to minimise this
        
        fakeA = self.poolA.query(fakeA)

        fakeAscore = self.discA(fakeA)
        return self.gan_loss_fn(realAscore, torch.ones_like(realAscore)) + \
                    self.gan_loss_fn(fakeAscore, torch.zeros_like(fakeAscore))

    def disc_loss_B(self, realBscore, fakeB):
        # Same as gan_loss_B, except targets are swapped and fakeB is randomly replaced from a buffer
        # Discriminator aims to minimise this

        fakeB = self.poolB.query(fakeB)
        
        fakeBscore = self.discB(fakeB)
        return self.gan_loss_fn(realBscore, torch.ones_like(realBscore)) + \
                    self.gan_loss_fn(fakeBscore, torch.zeros_like(fakeBscore))
    
    def disc_loss(self, realAscore, fakeA, realBscore, fakeB):
        return self.disc_loss_A(realAscore, fakeA) + self.disc_loss_B(realBscore, fakeB)


    def gan_loss_A(self, realAscore, fakeAscore):
        return self.gan_loss_fn(realAscore, torch.zeros_like(realAscore)) + \
                    self.gan_loss_fn(fakeAscore, torch.ones_like(fakeAscore))

    def gan_loss_B(self, realBscore, fakeBscore):
        return self.gan_loss_fn(realBscore, torch.zeros_like(realBscore)) + \
                    self.gan_loss_fn(fakeBscore, torch.ones_like(fakeBscore))

    def gan_loss(self, realAscore, fakeAscore, realBscore, fakeBscore):
        return self.gan_loss_A(realAscore, fakeAscore) + self.gan_loss_B(realBscore, fakeBscore)


    def cycle_loss(self, realA, realA_regen, realB, realB_regen):
        return self.cycle_loss_fn(realA, realA_regen) + \
                    self.cycle_loss_fn(realB, realB_regen)
    
    def identity_loss(self, realA, fakeA, realB, fakeB):
        return self.identity_loss_fn(realA, fakeB) + \
                    self.identity_loss_fn(realB, fakeA)

    def total_loss(self, realA, realAscore, realA_regen, realB, realBscore, realB_regen, fakeA, fakeAscore, fakeB, fakeBscore):
        gan_loss = self.gan_loss(realAscore, fakeAscore, realBscore, fakeBscore)
        cycle_loss = self.cycle_loss(realA, realA_regen, realB, realB_regen)
        
        identity_loss = self.identity_loss(realA, fakeA, realB, fakeB)
        return gan_loss + LAMBDA * cycle_loss + (LAMBDA / 2) * identity_loss

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
        self.genF.eval()
        self.genG.eval()
        self.discA.eval()
        self.discB.eval()

    def train(self):
        self.genF.train()
        self.genG.train()
        self.discA.train()
        self.discB.train()
    
    def cuda(self):
        self.genF.to('cuda')
        self.genG.to('cuda')
        self.discA.to('cuda')
        self.discB.to('cuda')
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
        self.genF(torch.zeros((1,3,IMG_RES,IMG_RES), device='cuda'))
        self.genG(torch.zeros((1,3,IMG_RES,IMG_RES), device='cuda'))
        self.discA(torch.zeros((1,3,IMG_RES,IMG_RES), device='cuda'))
        self.discB(torch.zeros((1,3,IMG_RES,IMG_RES), device='cuda'))
