from model import CycleGAN
from utils import plot_images_with_scores, infer_type, Mean
import logging
from configparser import ConfigParser
import wandb
import time
import atexit
import os
import sys
from datasets import JPGDataset
from torch.utils.data import DataLoader
import torch
from random import sample
import argparse
import torchvision.transforms as T
# ---------------------------------------------------------------------------- #
#                                     SETUP                                    #
# ---------------------------------------------------------------------------- #


# ---------------------------------- CONFIG ---------------------------------- #

config = ConfigParser({'INIT_WEIGHTS_WANDB_ARTIFACT':""})

config.read('config.ini')

parser = argparse.ArgumentParser()
parser.add_argument('data_dir')
parser.add_argument('--verbose','-v',action='store_true')
args = parser.parse_args()

WANDB_PROJECT_NAME = config.get('settings','WANDB_PROJECT_NAME')
WANDB_USER = config.get('settings', 'WANDB_USER')
LOG_FILE_NAME = config.get('settings', 'LOG_FILE_NAME')
LOG_FILE = f"./logs/{LOG_FILE_NAME}"

IMG_RES = config.getint('params', 'IMG_RES')
BATCH_SIZE = config.getint('params','BATCH_SIZE')
NUM_EPOCHS = config.getint('params', 'NUM_EPOCHS')
GEN_ARCHITECTURE = config.get('params', 'GEN_ARCHITECTURE')
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
if not os.path.isdir(CKPT_DIR):
    os.makedirs(CKPT_DIR)

logging.basicConfig(filename=LOG_FILE,
                    level=logging.DEBUG,
                    format = '%(asctime)s:%(levelname)s:%(name)s:%(message)s',
                    filemode='w')

# Remove annoying matplotlib.font_manager and PIL logs
logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
logging.getLogger('PIL').setLevel(logging.WARNING)

if args.verbose:
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

logging.info(f"Num GPUs: {torch.cuda.device_count()}")

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
    final_path = f"{CKPT_DIR}/final.pt"
    torch.save(model.ckpt(), final_path)

    artifact = wandb.Artifact(f"ckpt_final", type='model')

    artifact.add_file(final_path)

    wandb.log_artifact(artifact)


    # -------------------------------- Saving logs ------------------------------- #
    wandb.save(LOG_FILE)

    logging.info('Finished Training!')

atexit.register(on_exit)

# ---------------------------------------------------------------------------- #
#                        DATA LOADING AND PREPROCESSING                        #
# ---------------------------------------------------------------------------- #


photo_dataset = JPGDataset(f'{args.data_dir}/photo_jpg')
monet_dataset = JPGDataset(f'{args.data_dir}/monet_jpg')
augmented_photo_dataset = JPGDataset(f'{args.data_dir}/photo_jpg', augment=True)
augmented_monet_dataset = JPGDataset(f'{args.data_dir}/monet_jpg', augment=True)


setA = DataLoader(photo_dataset, batch_size=BATCH_SIZE, shuffle=True)
setB = DataLoader(monet_dataset, batch_size=BATCH_SIZE, shuffle=True)
augmented_setA = DataLoader(augmented_photo_dataset, batch_size=BATCH_SIZE, shuffle=True)
augmented_setB = DataLoader(augmented_monet_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Sample images to log later
fixed_sampleA = torch.stack([photo_dataset[idx] for idx in range(IMG_FIXED_LOG_NUM)])
fixed_sampleB = torch.stack([monet_dataset[idx] for idx in range(IMG_FIXED_LOG_NUM)])

# ---------------------------------------------------------------------------- #
#                                TRAINING SETUP                                #
# ---------------------------------------------------------------------------- #

# ------------------------------ CREATING MODEL ------------------------------ #
model = CycleGAN(GAN_LOSS_FN, generator=GEN_ARCHITECTURE, n_resblocks=N_RES_BLOCKS).cuda()

model.init_params()

with open('./logs/gen_architecture.txt', 'w') as f:
    f.write(model.genF.__repr__())
with open('./logs/disc_architecture.txt', 'w') as f:
    f.write(model.discA.__repr__())
wandb.watch(model.modules(), log='all')

# -------------------------------- OPTIMIZERS -------------------------------- #

disc_opt = torch.optim.Adam(model.get_disc_parameters(), lr=DISC_LR, betas=(0.5,0.999))
gen_opt = torch.optim.Adam(model.get_gen_parameters(), lr=GEN_LR, betas=(0.5,0.999))

def lr_lambda(epoch):
    len_decay_phase = NUM_EPOCHS - LR_DECAY_EPOCH + 1.0
    curr_decay_step = max(0, epoch - LR_DECAY_EPOCH + 1.0)
    val = 1.0 - curr_decay_step / len_decay_phase
    return max(0.0, val)
disc_scheduler = torch.optim.lr_scheduler.LambdaLR(disc_opt, lr_lambda)
gen_scheduler = torch.optim.lr_scheduler.LambdaLR(gen_opt, lr_lambda)

# ------------------------------ INITIALIZATION ------------------------------ #

def init_weights(m):
    if isinstance(m, (torch.nn.Conv2d, torch.nn.ConvTranspose2d, torch.nn.InstanceNorm2d)):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0.0)
model.apply(init_weights)

# ---------------------------------------------------------------------------- #
#                                 TRAINING STEP                                #
# ---------------------------------------------------------------------------- #

def train_one_epoch(step):
    logging.info(f'Training Epoch {step}...')
    model.train()

    start_time = time.perf_counter()

    disc_loss_metric = Mean()
    gan_loss_metric = Mean()
    cycle_loss_metric = Mean()
    identity_loss_metric = Mean()
    loss_metric = Mean()

    # --------------------------------- TRAINING --------------------------------- #
    # setB_iter = iter(augmented_setB)
    # for real_A in augmented_setA:
    #     try:
    #         real_B = next(setB_iter)
    #     except StopIteration:
    #         setB_iter = iter(augmented_setB)
    #         real_B = next(setB_iter)
    for real_A, real_B in zip(augmented_setA, augmented_setB):
        real_A = real_A.to('cuda')
        real_B = real_B.to('cuda')

        # Train discriminator
        disc_opt.zero_grad()
        with torch.no_grad():
            fake_B = model.genF(real_A)
            fake_A = model.genG(real_B)

        disc_loss = model.disc_loss(real_A, fake_A, real_B, fake_B) 
        disc_loss.backward()
        disc_opt.step()

        # Train generator
        gen_opt.zero_grad()
        fake_B = model.genF(real_A)
        fake_A = model.genG(real_B)

        gan_loss = model.gan_loss(fake_A, fake_B)
        cycle_loss = model.cycle_loss(real_A, fake_A, real_B, fake_B)
        identity_loss = model.identity_loss(real_A, real_B)

        loss = LAMBDA * cycle_loss + (LAMBDA/2) * identity_loss + gan_loss 

        loss.backward()
        gen_opt.step()

        disc_loss_metric(disc_loss.item())
        gan_loss_metric(gan_loss.item())     
        cycle_loss_metric(cycle_loss.item())
        identity_loss_metric(identity_loss.item())
        loss_metric(loss.item())
    # ---------------------------------- LOGGING --------------------------------- #
    time_taken = time.perf_counter() - start_time
    logging.info(f"Completed epoch {step}, time = {time_taken:.0f}s")
    wandb.log({
        'disc_loss': disc_loss_metric.result(),
        'gan_loss': gan_loss_metric.result(),
        'cycle_loss': cycle_loss_metric.result(),
        'identity_loss': identity_loss_metric.result(),
        'loss': loss_metric.result(),
        'time_per_epoch': time_taken,
        'learning_rate': disc_scheduler.get_last_lr()[0]
    }, step=step)
    
    disc_scheduler.step()
    gen_scheduler.step()


# ---------------------------------------------------------------------------- #
#                                 TRAINING LOOP                                #
# ---------------------------------------------------------------------------- #
logging.info('Starting Training ...')
for step in range(1, NUM_EPOCHS+1):
    train_one_epoch(step)

    # ------------------------------ Logging images ------------------------------ #
    model.eval()
    if step % IMG_LOG_FREQ == 0 or step == NUM_EPOCHS or step == 1:
        logging.info(f'Logging Images...')
        
        rand_sampleA_idx = sample(list(range(len(photo_dataset))), IMG_RANDOM_LOG_NUM)
        rand_sampleB_idx = sample(list(range(len(monet_dataset))), IMG_RANDOM_LOG_NUM)
        rand_sampleA = torch.stack([photo_dataset[idx] for idx in rand_sampleA_idx])
        rand_sampleB = torch.stack([monet_dataset[idx] for idx in rand_sampleB_idx])

        photo = torch.concat([fixed_sampleA,rand_sampleA], dim=0).to('cuda')
        monet = torch.concat([fixed_sampleB,rand_sampleB], dim=0).to('cuda')
        
        photo_fig = plot_images_with_scores(photo, model, which_set='A')
        monet_fig = plot_images_with_scores(monet, model, which_set='B')

        wandb.log({
            'Photo': photo_fig,
            'Monet': monet_fig
        })

    # ---------------------------- Creating checkpoint --------------------------- #
    if step % CKPT_FREQ == 0:
        logging.info('Creating checkpoint...')
        ckpt_path = f"{CKPT_DIR}/epoch{step}.pt"
        torch.save(model.ckpt(), ckpt_path)

        artifact = wandb.Artifact(f"ckpt_epoch{step}", type='model')

        artifact.add_file(ckpt_path)

        wandb.log_artifact(artifact)

