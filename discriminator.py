import tensorflow as tf
import os
from dotenv import load_dotenv

load_dotenv()

PATCH_SIZE = int(os.getenv('PATCH_SIZE'))
BATCH_SIZE = int(os.getenv('BATCH_SIZE'))

class Discriminator(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.layers.Input((PATCH_SIZE, PATCH_SIZE, 3), batch_size = BATCH_SIZE))

        self.model.add(tf.keras.layers.Conv2D(64, kernel_size=(4,4), strides=2, padding='same'))
        self.model.add(tf.keras.layers.BatchNormalization(axis=[0,1]))
        self.model.add(tf.keras.layers.LeakyReLU(0.2))

        self.model.add(tf.keras.layers.Conv2D(128, kernel_size=(4,4), strides=2, padding='same'))
        self.model.add(tf.keras.layers.BatchNormalization(axis=[0,1]))
        self.model.add(tf.keras.layers.LeakyReLU(0.2))
        
        self.model.add(tf.keras.layers.Conv2D(256, kernel_size=(4,4), strides=2, padding='same'))
        self.model.add(tf.keras.layers.BatchNormalization(axis=[0,1]))
        self.model.add(tf.keras.layers.LeakyReLU(0.2))

        self.model.add(tf.keras.layers.Conv2D(512, kernel_size=(4,4), strides=2, padding='same'))
        self.model.add(tf.keras.layers.BatchNormalization(axis=[0,1]))
        self.model.add(tf.keras.layers.LeakyReLU(0.2))

        self.model.add(tf.keras.layers.Conv2D(1, kernel_size=(4,4), strides=1, padding='same'))

    def call(self, X):
        return self.model(X)


