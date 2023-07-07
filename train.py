import tensorflow as tf
import tensorflow_datasets as tfds
from model import CycleGAN
from utils import ImagePool
import logging
from configparser import ConfigParser
import wandb
import time
import matplotlib.pyplot as plt

# ---------------------------------- CONFIG ---------------------------------- #

config = ConfigParser(defaults={
    'IMG_LOG_FREQ': 1,
    'IMG_LOG_NUM': 5
})

config.read('config.ini')

BATCH_SIZE = config.getint('params','BATCH_SIZE')
NUM_EPOCHS = config.getint('params', 'NUM_EPOCHS')
IMG_LOG_FREQ = config.getint('settings', 'IMG_LOG_FREQ')
IMG_LOG_NUM = config.getint('settings', 'IMG_LOG_NUM')

# ------------------------------- LOGGING SETUP ------------------------------ #
logging.basicConfig(filename='train.log',
                    level=logging.DEBUG,
                    format = '%(asctime)s:%(levelname)s:%(name)s:%(message)s',
                    filemode='w')

logging.info(f"Num GPUs: {len(tf.config.list_physical_devices('GPU'))}")
wandb.init(
    project='CycleGAN-Monet',
    config=dict(map(
        lambda item: (item[0], (float(item[1]) if item[1].isnumeric() else item[1])),
        config.items('params'))
    )
)


# ---------------------------------------------------------------------------- #
#                        DATA LOADING AND PREPROCESSING                        #
# ---------------------------------------------------------------------------- #

model = CycleGAN('bce', n_resblocks=6)

dataset = tfds.load('monet',batch_size=BATCH_SIZE)
setA, setB = dataset['photo'], dataset['monet']

def preprocess(X):
    return tf.cast(X['image'], dtype=tf.float32) / 255

setA = setA.map(preprocess).take(10)
setB = setB.map(preprocess).take(10)

# Sample images to log later
sampleA = setA.take(5)
sampleB = setB.take(5)

setA = setA.shuffle(500,seed=0,reshuffle_each_iteration=True)
setA = setB.shuffle(500,seed=0,reshuffle_each_iteration=True)

# ---------------------------------------------------------------------------- #
#                                TRAINING SETUP                                #
# ---------------------------------------------------------------------------- #

disc_opt = tf.optimizers.Adam(0.002)
gen_opt = tf.optimizers.Adam(0.002)
disc_opt.build(model.get_disc_trainable_variables())
gen_opt.build(model.get_gen_trainable_variables())

# ---------------------------------------------------------------------------- #
#                                 TRAINING STEP                                #
# ---------------------------------------------------------------------------- #

def train_one_epoch(step):
    start_time = time.perf_counter()

    disc_loss_metric = tf.metrics.Mean("disc_loss")
    gan_loss_metric = tf.metrics.Mean("gan_loss")
    cycle_loss_metric = tf.metrics.Mean("cycle_loss")
    loss_metric = tf.metrics.Mean("loss")

    # --------------------------------- TRAINING --------------------------------- #
    for imgA, imgB in zip(setA, setB):
        
        with tf.GradientTape() as disc_tape, tf.GradientTape() as gen_tape:
            realA, realAscore, fakeB, fakeBscore, realA_regen = model.forward_A(imgA)
            realB, realBscore, fakeA, fakeAscore, realB_regen = model.forward_B(imgB)

            disc_loss = model.disc_loss(realAscore, fakeA, realBscore, fakeB)

            gan_loss = model.gan_loss(realAscore, fakeAscore, realBscore, fakeBscore)
            cycle_loss = model.cycle_loss(realA, realA_regen, realB, realB_regen)
            loss = cycle_loss + gan_loss
            
        disc_grad = disc_tape.gradient(disc_loss, model.get_disc_trainable_variables())
        gen_grad = gen_tape.gradient(loss, model.get_gen_trainable_variables())

        disc_opt.apply_gradients(zip(disc_grad, model.get_disc_trainable_variables()))
        gen_opt.apply_gradients(zip(gen_grad, model.get_gen_trainable_variables()))

        disc_loss_metric(disc_loss)
        gan_loss_metric(gan_loss)     
        cycle_loss_metric(cycle_loss)
        loss_metric(loss)


    # ---------------------------------- LOGGING --------------------------------- #
    wandb.log({
        'disc_loss': disc_loss_metric.result(),
        'gan_loss': gan_loss_metric.result(),
        'cycle_loss': cycle_loss_metric.result(),
        'loss': loss_metric.result()
    }, step=step)

    logging.info(f"Completed epoch {step}, time = {time.perf_counter() - start_time}s")

    # TODO: log realA/fakeA/realA_regen triples with scores
    # TODO: log realB/fakeB/realB_regen triples with scores
    
# ---------------------------------------------------------------------------- #
#                                 TRAINING LOOP                                #
# ---------------------------------------------------------------------------- #

logging.info('Starting Training ...')
for step in range(1, NUM_EPOCHS+1):
    # train_one_epoch(step)

    # Log images
    fig, ax = plt.subplots(3,8,figsize=(3*8,8*8))
    rand_sampleA = setA.take(3)

    realscore = tf.reduce_mean(model.discA(img), axis=(1,2,3))
    fake = model.infer_A(img)
    fakescore = tf.reduce_mean(model.discB(img), axis=(1,2,3))
    regen = model.infer_B(img)
    regenscore = tf.reduce_mean(model.discA(regen), axis=(1,2,3))

    for idx, img in enumerate(sampleA.concatenate(rand_sampleA)):
        realscore = model.discA(img)
        ax[0,idx].imshow(img[idx,:,:,:])
        ax[0,idx].set_title(f"Score: {realscore[idx]:.2}")
        ax[0,idx].axis('off')

        out = model.infer_A(img)
        ax[1,idx].imshow(out[0,:,:,:])
        ax[0,idx].set_title(f"Score: {fakescore[idx]:.2}")
        ax[1,idx].axis('off')

        regen = model.infer_B(out)
        ax[2,idx].imshow(regen[0,:,:,:])
        ax[0,idx].set_title(f"Score: {regenscore[idx]:.2}")
        ax[2,idx].axis('off')

    fig.subplots_adjust(wspace=0,hspace=0)
    wandb.log({'Real Images':fig})
