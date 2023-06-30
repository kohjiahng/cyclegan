import tensorflow as tf
class ResBlock(tf.keras.layers.Layer):
    def __init__(self, kernel_size = 3, filters = 256):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters, kernel_size=kernel_size, strides=1, padding='same', activation='relu')
        self.conv2 = tf.keras.layers.Conv2D(filters, kernel_size=kernel_size, strides=1, padding='same', activation='relu')

    def call(self, X):
        x = self.conv1(X)
        y = self.conv2(x)
        return X+y
