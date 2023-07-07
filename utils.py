import random
import matplotlib.pyplot as plt
import tensorflow as tf

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

# -------------------- Plotting function for image logging ------------------- #
def plot_images_with_scores(images, model):
    fig, ax = plt.subplots(3,8,figsize=(images.shape[0]*8,3*8))

    realscore = tf.reduce_mean(model.discA(images), axis=(1,2,3))
    fake = model.infer_A(images)
    fakescore = tf.reduce_mean(model.discB(fake), axis=(1,2,3))
    regen = model.infer_B(fake)
    regenscore = tf.reduce_mean(model.discA(regen), axis=(1,2,3))
    for idx in range(images.shape[0]):
        ax[0,idx].imshow(images[idx,:,:,:])
        ax[0,idx].set_title(f"Score: {realscore[idx].numpy():.3}")
        ax[0,idx].axis('off')

        ax[1,idx].imshow(fake[0,:,:,:])
        ax[1,idx].set_title(f"Score: {fakescore[idx].numpy():.3}")
        ax[1,idx].axis('off')

        ax[2,idx].imshow(regen[0,:,:,:])
        ax[2,idx].set_title(f"Score: {regenscore[idx].numpy():.3}")
        ax[2,idx].axis('off')

    fig.subplots_adjust(wspace=0,hspace=0.1)

    return fig
