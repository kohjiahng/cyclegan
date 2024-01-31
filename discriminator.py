import tensorflow as tf
from configparser import ConfigParser

config = ConfigParser()
config.read('config.ini')

LAMBDA = config.getint('params', 'LAMBDA')
PATCH_SIZE = config.getint('params','PATCH_SIZE')
IMG_RES = config.getint('params','IMG_RES')
BATCH_SIZE = config.getint('params','BATCH_SIZE')
PATCHGAN_STRIDES = config.getint('params', 'PATCHGAN_STRIDES')
DISC_NOISE_STD = config.getfloat('params', 'DISC_NOISE_STD')

class Discriminator(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.layers.Input((IMG_RES, IMG_RES, 3)))

        # self.model.add(tf.keras.layers.GaussianNoise(DISC_NOISE_STD))

        self.model.add(tf.keras.layers.Conv2D(64, kernel_size=(4,4), strides=2, padding='same'))
        self.model.add(tf.keras.layers.GroupNormalization(groups=-1))
        self.model.add(tf.keras.layers.LeakyReLU(0.2))

        self.model.add(tf.keras.layers.Conv2D(128, kernel_size=(4,4), strides=2, padding='same'))
        self.model.add(tf.keras.layers.GroupNormalization(groups=-1))
        self.model.add(tf.keras.layers.LeakyReLU(0.2))
        
        self.model.add(tf.keras.layers.Conv2D(256, kernel_size=(4,4), strides=2, padding='same'))
        self.model.add(tf.keras.layers.GroupNormalization(groups=-1))
        self.model.add(tf.keras.layers.LeakyReLU(0.2))

        self.model.add(tf.keras.layers.Conv2D(512, kernel_size=(4,4), strides=2, padding='same'))
        self.model.add(tf.keras.layers.GroupNormalization(groups=-1))
        self.model.add(tf.keras.layers.LeakyReLU(0.2))

        self.model.add(tf.keras.layers.Conv2D(1, kernel_size=(4,4), strides=1, padding='same'))
        
        # Output is (16,16,1)

    def call(self, X):
        return self.model(X)
    
        N = X.shape[0] # number of images
        patches = tf.image.extract_patches( # Extracts patches from X
            X,
            [1, PATCH_SIZE, PATCH_SIZE, 1],
            strides=[1,PATCHGAN_STRIDES,PATCHGAN_STRIDES,1], # Strides to avoid OOM
            rates=[1,1,1,1],
            padding='VALID'
        )
        patches = tf.reshape(patches, (-1,PATCH_SIZE,PATCH_SIZE,3))
        output = self.model(patches)
        output = tf.reshape(output, (N, -1))

        return tf.reduce_mean(output, axis = 1)


