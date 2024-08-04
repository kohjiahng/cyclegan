import tensorflow as tf
import tensorflow_datasets as tfds
from model import CycleGAN
from utils import plot_images_with_scores, infer_type
import logging
from configparser import ConfigParser
import wandb
import time
import atexit
import os
from augment import get_data_augmentation

# ---------------------------------------------------------------------------- #
#                                     SETUP                                    #
# ---------------------------------------------------------------------------- #


# ---------------------------------- CONFIG ---------------------------------- #

config = ConfigParser({'INIT_WEIGHTS_WANDB_ARTIFACT':""})

config.read('config.ini')

WANDB_PROJECT_NAME = config.get('settings','WANDB_PROJECT_NAME')
WANDB_USER = config.get('settings', 'WANDB_USER')
LOG_FILE_NAME = config.get('settings', 'LOG_FILE_NAME')
LOG_FILE = f"./logs/{LOG_FILE_NAME}"

IMG_RES = config.getint('params', 'IMG_RES')
BATCH_SIZE = config.getint('params','BATCH_SIZE')
NUM_EPOCHS = config.getint('params', 'NUM_EPOCHS')
N_RES_BLOCKS = config.getint('params', 'N_RES_BLOCKS')
DISC_LR = config.getfloat('params', 'DISC_LR')
GEN_LR = config.getfloat('params', 'GEN_LR')
LR_DECAY_EPOCH = config.getint('params', 'LR_DECAY_EPOCH')
GAN_LOSS_FN = config.get('params', 'GAN_LOSS_FN')
LAMBDA = config.getfloat('params', 'LAMBDA')
IMG_LOG_FREQ = config.getint('settings', 'IMG_LOG_FREQ')
IMG_FIXED_LOG_NUM = config.getint('settings', 'IMG_FIXED_LOG_NUM')
IMG_RANDOM_LOG_NUM = config.getint('settings', 'IMG_RANDOM_LOG_NUM')

CKPT_FREQ = config.getint('settings', 'CKPT_FREQ')
CKPT_DIR = config.get('settings', 'CKPT_DIR')

INIT_WEIGHTS_WANDB_ARTIFACT = config.get('params', 'INIT_WEIGHTS_WANDB_ARTIFACT')
LOAD_WEIGHTS_GEN = config.getboolean('params','LOAD_WEIGHTS_GEN')
LOAD_WEIGHTS_DISC = config.getboolean('params', 'LOAD_WEIGHTS_DISC')

GEN_TRAINING_ONLY = config.getboolean('params', 'GEN_TRAINING_ONLY')

# ------------------------------- LOGGING SETUP ------------------------------ #

# Creating log folder and file if not exist
if not os.path.isdir('./logs'):
    os.makedirs('./logs')
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, 'w') as file:
        pass

logging.basicConfig(filename=LOG_FILE,
                    level=logging.DEBUG,
                    format = '%(asctime)s:%(levelname)s:%(name)s:%(message)s',
                    filemode='w')

# Remove annoying matplotlib.font_manager and PIL logs
logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
logging.getLogger('PIL').setLevel(logging.WARNING)

logging.info(f"Num GPUs: {len(tf.config.list_physical_devices('GPU'))}")
if GEN_TRAINING_ONLY:
    logging.info('Only training generator')
else:
    logging.info('Training both generator and discriminator')

wandb.init(
    project=WANDB_PROJECT_NAME,
    config=dict(map(
        lambda item: (item[0], infer_type(item[1])), # infer type of item[1]
        config.items('params'))
    ),
    settings=wandb.Settings(code_dir='.') # Code logging
)

# ---------------------------------- CLEANUP --------------------------------- #
def on_exit():
    # --------------------------- Saving final weights --------------------------- #
    logging.info('Saving final weights...')
    final_dir = f"{CKPT_DIR}/final"
    os.makedirs(final_dir, exist_ok=True)
    model.save_weights_separate(final_dir)

    wandb.save(final_dir)

    artifact = wandb.Artifact(f"ckpt_final", type='model')
    artifact.add_dir(final_dir)
    wandb.log_artifact(artifact)

    # -------------------------------- Saving logs ------------------------------- #
    wandb.save(LOG_FILE)

    logging.info('Finished Training!')


atexit.register(on_exit)

# ---------------------------------------------------------------------------- #
#                        DATA LOADING AND PREPROCESSING                        #
# ---------------------------------------------------------------------------- #


dataset = tfds.load('monet',batch_size=BATCH_SIZE)
setA, setB = dataset['photo'], dataset['monet']

def extract_image(X):
    return tf.cast(X['image'], dtype=tf.float32)

def scale(X):
    return X/127.5-1 # Scale to [-1,1]

setA = setA.map(extract_image)
setB = setB.map(extract_image)

data_augmentation = get_data_augmentation()
augmented_setA = setA.map(data_augmentation)
augmented_setB = setB.map(data_augmentation)

setA = setA.map(scale)
augmented_setA = augmented_setA.map(scale)
setB = setB.map(scale)
augmented_setB = augmented_setB.map(scale)

# Sample images to log later
sampleA = setA.take(IMG_FIXED_LOG_NUM)
sampleB = setB.take(IMG_FIXED_LOG_NUM)

setA = setA.shuffle(500,seed=0,reshuffle_each_iteration=True)
setB = setB.shuffle(500,seed=0,reshuffle_each_iteration=True)
augmented_setA = augmented_setA.shuffle(500,seed=0,reshuffle_each_iteration=True)
augmented_setB = augmented_setB.shuffle(500,seed=0,reshuffle_each_iteration=True)


# ---------------------------------------------------------------------------- #
#                                TRAINING SETUP                                #
# ---------------------------------------------------------------------------- #

# ------------------------------ CREATING MODEL ------------------------------ #
model = CycleGAN(GAN_LOSS_FN, n_resblocks=N_RES_BLOCKS)

if INIT_WEIGHTS_WANDB_ARTIFACT: 
    logging.info(f"Downloading weight artifact from {INIT_WEIGHTS_WANDB_ARTIFACT}...")

    artifact = wandb.use_artifact(INIT_WEIGHTS_WANDB_ARTIFACT)
    weight_dir = artifact.download()

    if LOAD_WEIGHTS_GEN:
        model.load_gen_weights(weight_dir)
        logging.info(f"Loaded generator weights!")

    if LOAD_WEIGHTS_DISC:
        model.load_disc_weights(weight_dir)
        logging.info(f"Loaded discriminator weights!")

# -------------------------------- OPTIMIZERS -------------------------------- #
class LinearDecaySchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    # Constant until decay_epoch, then linear to 0 until n_epoch
    def __init__(self, initial_learning_rate, decay_epoch, n_epoch):
        self.initial_learning_rate = initial_learning_rate
        self.decay_epoch = decay_epoch
        self.n_epoch = n_epoch
        self.decay = self.initial_learning_rate / max(1,n_epoch - decay_epoch)

    def __call__(self, step):
        step = tf.cast(step, dtype=tf.float32)
        if self.n_epoch < self.decay_epoch:
            return self.initial_learning_rate
        lr = self.initial_learning_rate - (step - self.decay_epoch) * self.decay
        return max(0, min(lr, self.initial_learning_rate))

disc_opt = tf.optimizers.Adam(LinearDecaySchedule(DISC_LR, LR_DECAY_EPOCH, NUM_EPOCHS))
gen_opt = tf.optimizers.Adam(LinearDecaySchedule(GEN_LR, LR_DECAY_EPOCH, NUM_EPOCHS))
disc_opt.build(model.get_disc_trainable_variables())
gen_opt.build(model.get_gen_trainable_variables())

# ---------------------------------------------------------------------------- #
#                                 TRAINING STEP                                #
# ---------------------------------------------------------------------------- #

def train_one_epoch(step):
    logging.info(f'Training Epoch {step}...')

    start_time = time.perf_counter()

    disc_loss_metric = tf.metrics.Mean("disc_loss")
    gan_loss_metric = tf.metrics.Mean("gan_loss")
    cycle_loss_metric = tf.metrics.Mean("cycle_loss")
    identity_loss_metric = tf.metrics.Mean("identity_loss")
    loss_metric = tf.metrics.Mean("loss")

    # --------------------------------- TRAINING --------------------------------- #
    for imgA, imgB in zip(augmented_setA, augmented_setB):
        
        with tf.GradientTape() as disc_tape, tf.GradientTape() as gen_tape:
            realA, realAscore, fakeB, fakeBscore, realA_regen = model.forward_A(imgA)
            realB, realBscore, fakeA, fakeAscore, realB_regen = model.forward_B(imgB)

            if GEN_TRAINING_ONLY:
                disc_loss = 0
                gan_loss = 0
            else:
                disc_loss = model.disc_loss(realAscore, fakeA, realBscore, fakeB) 
                gan_loss = model.gan_loss(realAscore, fakeAscore, realBscore, fakeBscore)

            cycle_loss = model.cycle_loss(realA, realA_regen, realB, realB_regen)
            identity_loss = model.identity_loss(realA, fakeA, realB, fakeB)

            loss = 0*LAMBDA * cycle_loss + (LAMBDA/2) * identity_loss + gan_loss 
            
        # if step % 2 == 0:
        #     disc_grad = disc_tape.gradient(disc_loss, model.get_disc_trainable_variables())
        #     disc_opt.apply_gradients(zip(disc_grad, model.get_disc_trainable_variables()))
        # else:
        gen_grad = gen_tape.gradient(loss, model.get_gen_trainable_variables())
        gen_opt.apply_gradients(zip(gen_grad, model.get_gen_trainable_variables()))


        disc_loss_metric(disc_loss)
        gan_loss_metric(gan_loss)     
        cycle_loss_metric(cycle_loss)
        identity_loss_metric(identity_loss)
        loss_metric(loss)

    # ---------------------------------- LOGGING --------------------------------- #
    time_taken = time.perf_counter() - start_time
    logging.info(f"Completed epoch {step}, time = {time_taken:.0f}s")
    wandb.log({
        'disc_loss': disc_loss_metric.result(),
        'gan_loss': gan_loss_metric.result(),
        'cycle_loss': cycle_loss_metric.result(),
        'identity_loss': identity_loss_metric.result(),
        'loss': loss_metric.result(),
        'time_per_epoch': time_taken
    }, step=step)


# ---------------------------------------------------------------------------- #
#                                 TRAINING LOOP                                #
# ---------------------------------------------------------------------------- #
logging.info('Starting Training ...')
for step in range(1, NUM_EPOCHS+1):
    train_one_epoch(step)

    # ------------------------------ Logging images ------------------------------ #

    if step % IMG_LOG_FREQ == 0 or step == NUM_EPOCHS or step == 1:
        logging.info(f'Logging Images...')

        rand_sampleA = setA.take(IMG_RANDOM_LOG_NUM) 
        rand_sampleB = setB.take(IMG_RANDOM_LOG_NUM)

        photo = tf.concat(list(sampleA.concatenate(rand_sampleA)), axis=0)
        monet = tf.concat(list(sampleB.concatenate(rand_sampleB)), axis=0)
        
        photo_fig = plot_images_with_scores(photo, model, which_set='A')
        monet_fig = plot_images_with_scores(monet, model, which_set='B')

        wandb.log({
            'Photo': photo_fig,
            'Monet': monet_fig
        })

    # ---------------------------- Creating checkpoint --------------------------- #
    if step % CKPT_FREQ == 0:
        logging.info('Creating checkpoint...')
        ckpt_dir = f"{CKPT_DIR}/epoch{step}"
        os.makedirs(ckpt_dir, exist_ok=True)

        model.save_weights_separate(ckpt_dir)
        wandb.save(ckpt_dir)

        artifact = wandb.Artifact(f"ckpt_epoch{step}", type='model')

        artifact.add_dir(ckpt_dir)

        wandb.log_artifact(artifact)

