import tensorflow as tf
from resblock import ResBlock
from configparser import ConfigParser

config = ConfigParser()
config.read('config.ini')

IMG_RES = config.getint('params','IMG_RES')
BATCH_SIZE = config.getint('params', 'BATCH_SIZE')

class Generator(tf.keras.Model):
    def __init__(self, n_resblocks):
        super().__init__()
        self.model = tf.keras.Sequential()

        self.model.add(tf.keras.layers.Input((IMG_RES, IMG_RES, 3)))

        self.model.add(tf.keras.layers.Conv2D(64,kernel_size=(7,7),strides=1, padding='same'))
        self.model.add(tf.keras.layers.GroupNormalization(groups=-1))
        self.model.add(tf.keras.layers.ReLU())
        assert self.model.output_shape == (None, IMG_RES, IMG_RES, 64)

        self.model.add(tf.keras.layers.Conv2D(128, kernel_size=(3,3),strides=2, padding='same'))
        self.model.add(tf.keras.layers.GroupNormalization(groups=-1))
        self.model.add(tf.keras.layers.ReLU())
        assert self.model.output_shape == (None, IMG_RES//2, IMG_RES//2, 128)

        self.model.add(tf.keras.layers.Conv2D(256, kernel_size=(3,3),strides=2, padding='same'))
        self.model.add(tf.keras.layers.GroupNormalization(groups=-1))
        self.model.add(tf.keras.layers.ReLU())

        assert self.model.output_shape == (None, IMG_RES//4, IMG_RES//4, 256)

        for _ in range(n_resblocks):
            self.model.add(ResBlock(3, 256))

        assert self.model.output_shape == (None, IMG_RES//4, IMG_RES//4, 256)

        self.model.add(tf.keras.layers.Conv2DTranspose(128, kernel_size=(3,3), strides=2, padding='same'))
        self.model.add(tf.keras.layers.GroupNormalization(groups=-1))
        self.model.add(tf.keras.layers.ReLU())

        assert self.model.output_shape == (None, IMG_RES//2, IMG_RES//2, 128)

        self.model.add(tf.keras.layers.Conv2DTranspose(64, kernel_size=(3,3), strides=2, padding='same'))
        self.model.add(tf.keras.layers.GroupNormalization(groups=-1))
        self.model.add(tf.keras.layers.ReLU())

        assert self.model.output_shape == (None, IMG_RES, IMG_RES, 64)

        self.model.add(tf.keras.layers.Conv2D(3, kernel_size=(7,7), strides=1, padding='same'))
        self.model.add(tf.keras.layers.GroupNormalization(groups=-1))
        self.model.add(tf.keras.layers.ReLU())

        assert self.model.output_shape == (None, IMG_RES, IMG_RES, 3)

    def call(self, X):
        return self.model(X)

if __name__ == '__main__':
    gen = Generator(6)