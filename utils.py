import random
import matplotlib.pyplot as plt
import torch
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

def channel_first(x):
    return torch.permute(x, (0,3,1,2))
def channel_last(x):
    return torch.permute(x, (0,2,3,1))

# -------------------- Plotting function for image logging ------------------- #

def plot_images_with_scores(images, model, which_set):
    model.eval()
    with torch.no_grad():

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

    images = channel_last(images).cpu()
    fake = channel_last(fake).cpu()
    regen = channel_last(regen).cpu()

    fig, ax = plt.subplots(3,8,figsize=(images.shape[0]*8,3*8))
    for idx in range(images.shape[0]):
        ax[0,idx].imshow((images[idx,:,:,:]+1)/2)
        ax[0,idx].set_title(f"Score: {realscore[idx]:.3}")
        ax[0,idx].axis('off')

        ax[1,idx].imshow((fake[idx,:,:,:]+1)/2)
        ax[1,idx].set_title(f"Score: {fakescore[idx]:.3}")
        ax[1,idx].axis('off')

        ax[2,idx].imshow((regen[idx,:,:,:]+1)/2)
        ax[2,idx].set_title(f"Score: {regenscore[idx]:.3}")
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

class Mean():
    def __init__(self):
        self.count = 0
        self.sum = 0
    def __call__(self,val):
        self.count += 1
        self.sum += val
    def result(self):
        return self.sum / self.count