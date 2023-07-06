import tensorflow as tf
import tensorflow_datasets as tfds
from model import CycleGAN
from utils import ImagePool
import logging
from configparser import ConfigParser

# ---------------------------------- CONFIG ---------------------------------- #

config = ConfigParser()
config.read('config.ini')

BATCH_SIZE = int(config.get('main','BATCH_SIZE'))

# ------------------------------- LOGGING SETUP ------------------------------ #
logging.basicConfig(filename='train.log',
                    level=logging.DEBUG,
                    format = '%(asctime)s:%(levelname)s:%(name)s:%(message)s',
                    filemode='w')


logging.info(f"Num GPUs: {len(tf.config.list_physical_devices('GPU'))}")

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
#                                   TRAINING                                   #
# ---------------------------------------------------------------------------- #

# ------------------------------ TRAINING SETUP ------------------------------ #
poolA, poolB = ImagePool(50), ImagePool(50)
disc_opt = tf.optimizers.Adam(0.002)
gen_opt = tf.optimizers.Adam(0.002)
disc_opt.build(model.get_disc_trainable_variables())
gen_opt.build(model.get_gen_trainable_variables())

# ------------------------------- TRAINING STEP ------------------------------ #
def train_one_epoch():
    for imgA, imgB in zip(setA, setB):
        imgA = poolA.query(imgA)
        imgB = poolB.query(imgB)
        
        with tf.GradientTape() as disc_tape, tf.GradientTape() as gen_tape:
            realA, realAscore, fakeB, fakeBscore, realA_regen = model.forward_A(imgA)
            realB, realBscore, fakeA, fakeAscore, realB_regen = model.forward_B(imgB)

            gan_loss = -model.gan_loss(realAscore, fakeAscore, realBscore, fakeBscore) # Negate as discriminator tries to maximise this value
            loss = model.cycle_loss(realA, realA_regen, realB, realB_regen) - gan_loss

        disc_grad = disc_tape.gradient(gan_loss, model.get_disc_trainable_variables())
        gen_grad = gen_tape.gradient(loss, model.get_gen_trainable_variables())

        disc_opt.apply_gradients(zip(disc_grad, model.get_disc_trainable_variables()))
        gen_opt.apply_gradients(zip(gen_grad, model.get_gen_trainable_variables()))

        logging.info(loss.numpy())

# --------------------------------- TRAINING --------------------------------- #

logging.info('Starting Training ...')
for _ in range(10):
    train_one_epoch()