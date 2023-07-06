import tensorflow as tf
import numpy as np
from configparser import ConfigParser

config = ConfigParser()
config.read('config.ini')

LAMBDA = config.getint('main', 'LAMBDA')
PATCH_SIZE = config.getint('main','PATCH_SIZE')
BATCH_SIZE = config.getint('main','BATCH_SIZE')

class Discriminator(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.layers.Input((PATCH_SIZE, PATCH_SIZE, 3)))

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

    def call(self, X):
        patches = tf.image.extract_patches( # Extracts patches from X
            X,
            [1, PATCH_SIZE, PATCH_SIZE, 1],
            strides=[1,32,32,1], # Strides to avoid OOM
            rates=[1,1,1,1],
            padding='VALID'
        )
        patches = tf.reshape(patches, (-1,PATCH_SIZE,PATCH_SIZE,3))

        return self.model(patches)


