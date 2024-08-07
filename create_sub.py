from datasets import JPGDataset
from torch.utils.data import DataLoader
from model import CycleGAN
from configparser import ConfigParser
import wandb
from PIL import Image
import numpy as np
from utils import infer_type, channel_last
import logging
import os
import torch
import argparse
from generator import Generator

# ---------------------------------------------------------------------------- #
#                                     SETUP                                    #
# ---------------------------------------------------------------------------- #

# ---------------------------------- CONFIG ---------------------------------- #
parser = argparse.ArgumentParser()
parser.add_argument('data_dir')
args = parser.parse_args()
config = ConfigParser({'INIT_WEIGHTS_WANDB_ARTIFACT':""})

config.read('config.ini')

WANDB_USER = config.get("settings", 'WANDB_USER')
WANDB_PROJECT_NAME = config.get('settings','WANDB_PROJECT_NAME')
WANDB_ARTIFACT_NAME = config.get('submission', 'WANDB_ARTIFACT_NAME')
CKPT_FILE_NAME = config.get('submission', 'CKPT_FILE_NAME')
N_RES_BLOCKS = config.getint('params', 'N_RES_BLOCKS')
BATCH_SIZE = config.getint('submission', 'BATCH_SIZE')
LOG_FILE_NAME = config.get('submission', 'LOG_FILE_NAME')
LOG_FILE = f"./logs/{LOG_FILE_NAME}"

# ---------------------------------- LOGGING --------------------------------- #
if not os.path.isdir('./logs'):
    os.makedirs('./logs')
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, 'w') as file:
        pass

logging.basicConfig(filename=LOG_FILE,
                    level=logging.DEBUG,
                    format = '%(asctime)s:%(levelname)s:%(name)s:%(message)s',
                    filemode='w')

logging.info(f"Num GPUs: {torch.cuda.device_count()}")

# ------------------------------- DATA LOADING ------------------------------- #
photo_dataset = JPGDataset(f'{args.data_dir}/photo_jpg')

setA = DataLoader(photo_dataset, batch_size=BATCH_SIZE, shuffle=True)

api = wandb.Api()
artifact = api.artifact(f"{WANDB_USER}/{WANDB_PROJECT_NAME}/{WANDB_ARTIFACT_NAME}")
weight_dir = artifact.download()


genF = Generator(N_RES_BLOCKS).cuda()
genF.eval()

# Initalize parameter sizes (lazy modules)
genF(torch.zeros((1,3,256,256),device='cuda'))

ckpt = torch.load(f'{weight_dir}/{CKPT_FILE_NAME}')
genF.load_state_dict(ckpt['genF_state_dict'])

logging.info(f"Loaded generator weights, generating images...")

idx = 0
for imgA in setA:
    with torch.no_grad():
        generated = genF(imgA.cuda())
        generated = 255*(generated+1)/2
        generated = channel_last(generated)
    
    for img in generated.cpu():
        pil_img = Image.fromarray(np.uint8(img.numpy()))
        pil_img.save(f'./images/{idx}.jpg')
        idx += 1

        if idx % 1000 == 0:
            logging.info(f'{idx} images generated!')

logging.info(f"Finished generating {idx} images!")
