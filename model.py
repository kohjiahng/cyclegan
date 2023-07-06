import tensorflow as tf
from generator import Generator
from discriminator import Discriminator
from dotenv import load_dotenv
import os

load_dotenv()

LAMBDA = int(os.getenv('LAMBDA'))

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

    @tf.function
    def forward_A(self, X):
        '''
        Stores X in self.realA
        Passes X through self.discA and stores the output in self.realAscore
        Passes X through self.genF and stores the output in self.fakeB
        Passes self.fakeB through self.discB and stores the output in self.fakeBscore
        '''
        X = tf.cast(X, dtype=tf.float32)

        realA = X
        realAscore = self.discA(X)
        fakeB = self.genF(X)
        fakeBscore = self.discB(fakeB)
        realA_regen = self.genG(fakeB)

        return realA, realAscore, fakeB, fakeBscore, realA_regen

    @tf.function
    def forward_B(self, X):
        '''
        Stores X in self.realB
        Passes X through self.discB and stores the output in self.realBscore
        Passes X through self.genG and stores the output in self.fakeA
        Passes self.fakeA through self.discA and stores the output in self.fakeAscore
        Passes self.fakeA through self.genF and stores the output in self.realB_regen
        '''
        X = tf.cast(X, dtype=tf.float32)

        realB = X
        realBscore = self.discB(X)
        fakeA = self.genG(X)
        fakeAscore = self.discA(fakeA)
        realB_regen = self.genF(fakeA)

        return realB, realBscore, fakeA, fakeAscore, realB_regen 

    @tf.function
    def gan_loss_A(self, realAscore, fakeAscore):
        return self.gan_loss_fn(tf.zeros_like(realAscore), realAscore) + \
                    self.gan_loss_fn(tf.ones_like(fakeAscore), fakeAscore)

    @tf.function
    def gan_loss_B(self, realBscore, fakeBscore):
        return self.gan_loss_fn(tf.zeros_like(realBscore), realBscore) + \
                    self.gan_loss_fn(tf.ones_like(fakeBscore), fakeBscore)

    @tf.function
    def gan_loss(self, realAscore, fakeAscore, realBscore, fakeBscore):
        return self.gan_loss_A(realAscore, fakeAscore) + self.gan_loss_B(realBscore, fakeBscore)


    @tf.function
    def cycle_loss(self, realA, realA_regen, realB, realB_regen):
        return self.cycle_loss_fn(realA, realA_regen) + \
                    self.cycle_loss_fn(realB, realB_regen)
    
    @tf.function
    def total_loss(self, realA, realAscore, realA_regen, realB, realBscore, realB_regen, fakeA, fakeAscore, fakeB, fakeBscore):
        gan_loss = self.gan_loss(realAscore, fakeAscore, realBscore, fakeBscore)
        cycle_loss = self.cycle_loss(realA, realA_regen, realB, realB_regen)
        return gan_loss + LAMBDA * cycle_loss

    def infer_B(self, X): # A to B
        return self.genF(X)
    
    def infer_A(self, X): # B to A
        return self.genG(X)
    
    def get_disc_trainable_variables(self):
        return self.discA.trainable_variables + self.discB.trainable_variables
    
    def get_gen_trainable_variables(self):
        return self.genF.trainable_variables + self.genG.trainable_variables
