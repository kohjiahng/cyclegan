import random
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

# ------------------ ImagePool class for discriminator loss ------------------ #
class ImagePool:
    def __init__(self, pool_size):
        self.pool_size = pool_size
        self.images = []

    def query(self, image):
        '''
        Randomly choose to
        1. Return input
        2. Return random image from pool and replace it with input 
        '''

        if len(self.images) < self.pool_size:
            self.images.append(image)
            return image
        else:
            if random.randint(0,1) == 1:
                return image
            else:
                idx = random.randint(0,len(self.images)-1)
                result = self.images.pop(idx)
                self.images.append(image)
                return result


# ------------------ Reflection Padding to prevent artifacts ----------------- #
# https://stackoverflow.com/questions/70382008/what-and-how-to-pretend-these-artifacts-in-training-gan
# https://stackoverflow.com/questions/50677544/reflection-padding-conv2d

class ReflectionPadding2D(tf.keras.layers.Layer):
    def __init__(self, padding=(1, 1), **kwargs):
        self.padding = tuple(padding)
        self.input_spec = [tf.keras.layers.InputSpec(ndim=4)]
        super(ReflectionPadding2D, self).__init__(**kwargs)

    def compute_output_shape(self, s):
        """ If you are using "channels_last" configuration"""
        return (s[0], s[1] + 2 * self.padding[0], s[2] + 2 * self.padding[1], s[3])

    def call(self, x, mask=None):
        w_pad,h_pad = self.padding
        return tf.pad(x, [[0,0], [h_pad,h_pad], [w_pad,w_pad], [0,0] ], 'REFLECT')

# -------------------- Plotting function for image logging ------------------- #

def plot_images_with_scores(images, model, which_set):
    fig, ax = plt.subplots(3,8,figsize=(images.shape[0]*8,3*8))

    if which_set == 'A':    
        realscore = model.discA(images)
        fake = model.infer_B(images)
        fakescore = model.discB(fake)
        regen = model.infer_A(fake)
        regenscore = model.discA(regen)
    elif which_set == 'B':
        realscore = model.discB(images)
        fake = model.infer_A(images)
        fakescore = model.discA(fake)
        regen = model.infer_B(fake)
        regenscore = model.discB(regen)
    else:
        raise Exception('which_set must be "A" or "B" in plot_images_with_scores')

    for idx in range(images.shape[0]):
        ax[0,idx].imshow((images[idx,:,:,:]+1)/2)
        ax[0,idx].set_title(f"Score: {np.mean(realscore[idx]):.3}")
        ax[0,idx].axis('off')

        ax[1,idx].imshow((fake[idx,:,:,:]+1)/2)
        ax[1,idx].set_title(f"Score: {np.mean(fakescore[idx]):.3}")
        ax[1,idx].axis('off')

        ax[2,idx].imshow((regen[idx,:,:,:]+1)/2)
        ax[2,idx].set_title(f"Score: {np.mean(regenscore[idx]):.3}")
        ax[2,idx].axis('off')

    fig.subplots_adjust(wspace=0,hspace=0.1)

    return fig

# ------------ infer_type convenience function for config logging ------------ #
def infer_type(val):
    try:
        return int(val)
    except ValueError:
        try:
            return float(val)
        except ValueError:
            return val