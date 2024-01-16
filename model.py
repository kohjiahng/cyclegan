import tensorflow as tf
from utils import ImagePool
from generator import Generator
from discriminator import Discriminator
from configparser import ConfigParser

config = ConfigParser()
config.read('config.ini')

POOL_SIZE = config.getint('params', 'POOL_SIZE')
LAMBDA = config.getint('params', 'LAMBDA')
IMG_RES = config.getint('params', 'IMG_RES')

class CycleGAN(tf.keras.Model):
    def __init__(self, gan_loss_fn = 'bce', n_resblocks=6):
        '''
        gan_loss_fn: either 'mse' or 'bce'
        '''
        super().__init__()

        self.genF = Generator(n_resblocks)
        self.discB = Discriminator()

        self.genG = Generator(n_resblocks)
        self.discA = Discriminator()

        self.poolA = ImagePool(POOL_SIZE)
        self.poolB = ImagePool(POOL_SIZE)

        if gan_loss_fn == 'mse':
            self.gan_loss_fn = tf.keras.losses.MeanSquaredError() 
        elif gan_loss_fn == 'bce':
            self.gan_loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        else:
            raise NotImplementedError
        
        self.cycle_loss_fn = tf.keras.losses.MeanAbsoluteError()
        self.identity_loss_fn = tf.keras.losses.MeanAbsoluteError()

    # ---------------------------- TRAINING FUNCTIONS ---------------------------- #
    @tf.function
    def forward_A(self, X):
        realA = X
        realAscore = self.discA(X)
        fakeB = self.genF(X)
        fakeBscore = self.discB(fakeB)
        realA_regen = self.genG(fakeB)

        return realA, realAscore, fakeB, fakeBscore, realA_regen

    @tf.function
    def forward_B(self, X):
        realB = X
        realBscore = self.discB(X)
        fakeA = self.genG(X)
        fakeAscore = self.discA(fakeA)
        realB_regen = self.genF(fakeA)

        return realB, realBscore, fakeA, fakeAscore, realB_regen 

    # ---------------------------------- LOSSES ---------------------------------- #
    @tf.function
    def disc_loss_A(self, realAscore, fakeA):
        # Same as gan_loss_A, except targets are swapped and fakeA is randomly replaced from a buffer and targets are swapped
        # Discriminator aims to minimise this
        
        fakeA = self.poolA.query(fakeA)

        fakeAscore = self.discA(fakeA)
        return self.gan_loss_fn(tf.ones_like(realAscore), realAscore) + \
                    self.gan_loss_fn(tf.zeros_like(fakeAscore), fakeAscore)

    @tf.function
    def disc_loss_B(self, realBscore, fakeB):
        # Same as gan_loss_B, except targets are swapped and fakeB is randomly replaced from a buffer
        # Discriminator aims to minimise this

        fakeB = self.poolB.query(fakeB)
        
        fakeBscore = self.discB(fakeB)
        return self.gan_loss_fn(tf.ones_like(realBscore), realBscore) + \
                    self.gan_loss_fn(tf.zeros_like(fakeBscore), fakeBscore)
    
    @tf.function
    def disc_loss(self, realAscore, fakeA, realBscore, fakeB):
        return self.disc_loss_A(realAscore, fakeA) + self.disc_loss_B(realBscore, fakeB)


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
    def identity_loss(self, realA, fakeA, realB, fakeB):
        return self.identity_loss_fn(realA, fakeB) + \
                    self.identity_loss_fn(realB, fakeA)
    @tf.function
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
    
    def get_disc_trainable_variables(self):
        return self.discA.trainable_variables + self.discB.trainable_variables
    
    def get_gen_trainable_variables(self):
        return self.genF.trainable_variables + self.genG.trainable_variables
    
    def save_weights_separate(self, dir):
        self.discA.save_weights(f"{dir}/discA.h5")
        self.discB.save_weights(f"{dir}/discB.h5")
        self.genF.save_weights(f"{dir}/genF.h5")
        self.genG.save_weights(f"{dir}/genG.h5")
        return self
    
    def load_disc_weights(self, dir):
        self.discA.build((None, IMG_RES, IMG_RES, 3))
        self.discB.build((None, IMG_RES, IMG_RES, 3))

        self.discA.load_weights(f"{dir}/discA.h5")
        self.discB.load_weights(f"{dir}/discB.h5")
        return self

    def load_gen_weights(self, dir):
        self.genF.build((None, IMG_RES, IMG_RES, 3))
        self.genG.build((None, IMG_RES, IMG_RES, 3))

        self.genF.load_weights(f"{dir}/genF.h5") 
        self.genG.load_weights(f"{dir}/genG.h5")