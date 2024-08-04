import tensorflow as tf
import tensorflow_datasets as tfds
from model import CycleGAN
from configparser import ConfigParser
import wandb
from PIL import Image
import numpy as np
from utils import infer_type
import logging
import os

config = ConfigParser({'INIT_WEIGHTS_WANDB_ARTIFACT':""})

config.read('config.ini')

WANDB_USER = config.get("settings", 'WANDB_USER')
WANDB_PROJECT_NAME = config.get('settings','WANDB_PROJECT_NAME')
WANDB_ARTIFACT_NAME = config.get('submission', 'WANDB_ARTIFACT_NAME')
GAN_LOSS_FN = config.get('params', 'GAN_LOSS_FN')
N_RES_BLOCKS = config.getint('params', 'N_RES_BLOCKS')
BATCH_SIZE = config.getint('submission', 'BATCH_SIZE')
LOG_FILE_NAME = config.get('submission', 'LOG_FILE_NAME')
LOG_FILE = f"./logs/{LOG_FILE_NAME}"



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

logging.info(f"Num GPUs: {len(tf.config.list_physical_devices('GPU'))}")

dataset = tfds.load('monet',batch_size=BATCH_SIZE)
setA, setB = dataset['photo'], dataset['monet']

def extract_image(X):
    return tf.cast(X['image'], dtype=tf.float32)

def scale(X):
    return X/127.5-1 # Scale to [-1,1]

setA = setA.map(extract_image)
setB = setB.map(extract_image)

setA = setA.map(scale)
setB = setB.map(scale)


api = wandb.Api()
artifact = api.artifact(f"{WANDB_USER}/{WANDB_PROJECT_NAME}/{WANDB_ARTIFACT_NAME}")
weight_dir = artifact.download()

model = CycleGAN(GAN_LOSS_FN, n_resblocks=N_RES_BLOCKS)

model.load_gen_weights(weight_dir)
logging.info(f"Loaded generator weights!")

model.load_disc_weights(weight_dir)
logging.info(f"Loaded discriminator weights!")

idx = 0
for imgA in setA:
    genB = model.genF(imgA)
    genBscaled = 255 * (genB+1)/2 # 0 to 255
    
    for img in genBscaled:
        Image.fromarray(np.uint8(img.numpy())).save(f'./images/{idx}.jpg')
        idx += 1

        if idx % 1000 == 0:
            logging.info(f'{idx} images generated!')