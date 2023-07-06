import tensorflow as tf
import tensorflow_datasets as tfds
from model import CycleGAN
from utils import ImagePool
import logging
from configparser import ConfigParser
import wandb
import time

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
        lambda key, value: (key, (float(value) if value.isnumeric() else value)),
        config.items('params'))
    )
)


# ---------------------------------------------------------------------------- #
#                        DATA LOADING AND PREPROCESSING                        #
# ---------------------------------------------------------------------------- #

model = CycleGAN('bce', n_resblocks=6)

dataset = tfds.load('monet',batch_size=BATCH_SIZE,shuffle_files=True)
setA, setB = dataset['monet'], dataset['photo']

def preprocess(X):
    return tf.cast(X['image'], dtype=tf.float32) / 255

setA = setA.map(preprocess)
setB = setB.map(preprocess)

# ---------------------------------------------------------------------------- #
#                                TRAINING SETUP                                #
# ---------------------------------------------------------------------------- #

poolA, poolB = ImagePool(50), ImagePool(50)
disc_opt = tf.optimizers.Adam(0.002)
gen_opt = tf.optimizers.Adam(0.002)
disc_opt.build(model.get_disc_trainable_variables())
gen_opt.build(model.get_gen_trainable_variables())

# ---------------------------------------------------------------------------- #
#                                 TRAINING STEP                                #
# ---------------------------------------------------------------------------- #

def train_one_epoch(step):
    start_time = time.perf_counter()

    gan_loss_metric = tf.metrics.Mean("gan_loss")
    cycle_loss_metric = tf.metrics.Mean("cycle_loss")
    loss_metric = tf.metrics.Mean("loss")

    # --------------------------------- TRAINING --------------------------------- #
    for imgA, imgB in zip(setA, setB):
        imgA = poolA.query(imgA)
        imgB = poolB.query(imgB)
        
        with tf.GradientTape() as disc_tape, tf.GradientTape() as gen_tape:
            realA, realAscore, fakeB, fakeBscore, realA_regen = model.forward_A(imgA)
            realB, realBscore, fakeA, fakeAscore, realB_regen = model.forward_B(imgB)

            gan_loss = -model.gan_loss(realAscore, fakeAscore, realBscore, fakeBscore) # Negate as discriminator tries to maximise this value
            cycle_loss = model.cycle_loss(realA, realA_regen, realB, realB_regen)
            loss = cycle_loss - gan_loss
            
        disc_grad = disc_tape.gradient(gan_loss, model.get_disc_trainable_variables())
        gen_grad = gen_tape.gradient(loss, model.get_gen_trainable_variables())

        disc_opt.apply_gradients(zip(disc_grad, model.get_disc_trainable_variables()))
        gen_opt.apply_gradients(zip(gen_grad, model.get_gen_trainable_variables()))

        gan_loss_metric(-gan_loss)     
        cycle_loss_metric(cycle_loss)
        loss_metric(loss)

    # ---------------------------------- LOGGING --------------------------------- #
    wandb.log({
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
    train_one_epoch(step)