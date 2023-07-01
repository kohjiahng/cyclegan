import tensorflow as tf
from generator import Generator
from discriminator import Discriminator
from dotenv import load_dotenv

load_dotenv()

class CycleGAN:
    def __init__(self, gan_loss = 'bce', n_resblocks=6):
        '''
        gan_loss: either 'mse' or 'bce'
        '''
        self.genF = Generator(n_resblocks)
        self.discB = Discriminator()

        self.genG = Generator(n_resblocks)
        self.discA = Discriminator()

        if gan_loss == 'mse':
            self.gan_loss_fn = tf.keras.losses.MeanSquaredError() 
        elif gan_loss == 'bce':
            self.gan_loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        else:
            raise NotImplementedError
        
        self.cycle_loss_fn = tf.keras.losses.MeanAbsoluteError()

    def forward(self, X):
        '''
        Stores X in self.realA
        Passes X through self.discA and stores the output in self.realAscore
        Passes X through self.genF and stores the output in self.fakeB
        Passes self.fakeB through self.discB and stores the output in self.fakeBscore
        '''

        self.realA = X
        self.realAscore = self.discA(X)
        self.fakeB = self.genF(X)
        self.fakeBscore = self.discB(self.fakeB)
        self.realA_regen = self.genG(self.fakeB)

        return True

    def backward(self, X):
        '''
        Stores X in self.realB
        Passes X through self.discB and stores the output in self.realBscore
        Passes X through self.genG and stores the output in self.fakeA
        Passes self.fakeA through self.discA and stores the output in self.fakeAscore
        Passes self.fakeA through self.genF and stores the output in self.realB_regen
        '''

        self.realB = X
        self.realBscore = self.discB(X)
        self.fakeA = self.genG(X)
        self.fakeAscore = self.discA(self.fakeA)
        self.realB_regen = self.genF(self.fakeA)

        return True

    def gan_loss_A(self):
        return self.gan_loss_fn(True, self.realA) + \
                    self.gan_loss_fn(False, self.fakeA)

    def gan_loss_B(self):
        return self.gan_loss_fn(True, self.realB) + \
                    self.gan_loss_fn(False, self.fakeB)

    def cycle_loss(self):
        return self.cycle_loss_fn(self.realA, self.realA_regen) + \
                    self.cycle_loss_fn(self.realB, self.realB_regen)

    def infer_B(self, X): # A to B
        return self.genF(X)
    
    def infer_A(self, X): # B to A
        return self.genG(X)
