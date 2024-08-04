import tensorflow as tf

initializer = tf.keras.initializers.RandomNormal(mean=0.0,stddev=0.02)
class ResBlock(tf.keras.layers.Layer):
    def __init__(self, kernel_size = 3, filters = 256):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters, kernel_size=kernel_size, strides=1, padding='same', activation='relu', kernel_initializer=initializer, bias_initializer=initializer)
        self.conv2 = tf.keras.layers.Conv2D(filters, kernel_size=kernel_size, strides=1, padding='same', activation='relu',kernel_initializer=initializer, bias_initializer=initializer)

    def call(self, X):
        x = self.conv1(X)
        y = self.conv2(x)
        return X+y

class ConvLayerNormRelu(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides, leaky=False):
        super().__init__()
        self.conv = tf.keras.layers.Conv2D(filters, kernel_size=kernel_size, strides=strides,padding='same',kernel_initializer=initializer,bias_initializer=initializer)
        self.layernorm = tf.keras.layers.GroupNormalization(groups=-1)
        self.relu = tf.keras.layers.LeakyReLU(0.2) if leaky else tf.keras.layers.ReLU()
    def call(self, X):
        return self.relu(self.layernorm(self.conv(X)))

class ConvTransposeLayerNormRelu(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides):
        super().__init__()
        self.conv = tf.keras.layers.Conv2DTranspose(filters, kernel_size=kernel_size, strides=strides,padding='same',kernel_initializer=initializer,bias_initializer=initializer)
        self.layernorm = tf.keras.layers.GroupNormalization(groups=-1)
        self.relu = tf.keras.layers.ReLU(0.2)
    def call(self, X):
        return self.relu(self.layernorm(self.conv(X)))