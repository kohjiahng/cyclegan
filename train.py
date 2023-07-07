import tensorflow as tf
import tensorflow_datasets as tfds
from model import CycleGAN
from utils import plot_images_with_scores
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
IMG_FIXED_LOG_NUM = config.getint('settings', 'IMG_FIXED_LOG_NUM')
IMG_RANDOM_LOG_NUM = config.getint('settings', 'IMG_RANDOM_LOG_NUM')

# ------------------------------- LOGGING SETUP ------------------------------ #
logging.basicConfig(filename='train.log',
                    level=logging.DEBUG,
                    format = '%(asctime)s:%(levelname)s:%(name)s:%(message)s',
                    filemode='w')

# Remove annoying matplotlib.font-manager logs
logging.getLogger('matplotlib.font_manager').disabled = True

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

setA = setA.map(preprocess)
setB = setB.map(preprocess)

# Sample images to log later
sampleA = setA.take(IMG_FIXED_LOG_NUM)
sampleB = setB.take(IMG_FIXED_LOG_NUM)

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

# ---------------------------------------------------------------------------- #
#                                 TRAINING LOOP                                #
# ---------------------------------------------------------------------------- #
logging.info('Starting Training ...')
for step in range(1, NUM_EPOCHS+1):
    train_one_epoch(step)

    # ------------------------------ Logging images ------------------------------ #

    if (step - 1) % IMG_LOG_FREQ == 0:
        rand_sampleA = sampleA.take(IMG_RANDOM_LOG_NUM) 
        rand_sampleB = sampleB.take(IMG_RANDOM_LOG_NUM)

        photo = tf.concat(list(sampleA.concatenate(rand_sampleA)), axis=0)
        monet = tf.concat(list(sampleB.concatenate(rand_sampleB)), axis=0)
        
        photo_fig = plot_images_with_scores(photo, model)
        monet_fig = plot_images_with_scores(monet, model)

        wandb.log({
            'Photo': photo_fig,
            'Monet': monet_fig
        })
