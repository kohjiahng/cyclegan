import tensorflow as tf
from tensorflow.keras import layers

from configparser import ConfigParser

config = ConfigParser()
config.read('config.ini')

IMG_RES = config.getint('params','IMG_RES')

def get_data_augmentation():
    data_augmentation = tf.keras.Sequential([
        layers.Resizing(286,286),
        layers.RandomCrop(IMG_RES,IMG_RES),
        layers.RandomFlip(mode="horizontal"),
        layers.RandomContrast(0.2),
        layers.RandomBrightness(0.2)
    ])
    return data_augmentation