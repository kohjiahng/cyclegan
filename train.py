import tensorflow as tf
import tensorflow_datasets as tfds
from model import CycleGAN
from utils import ImagePool
import os
import logging
from dotenv import load_dotenv

load_dotenv()

BATCH_SIZE = int(os.getenv('BATCH_SIZE'))
logging.basicConfig(filename='file.log',
                    level=logging.DEBUG,
                    format = '%(asctime)s:%(levelname)s:%(name)s:%(message)s',
                    filemode='w')


logging.info(f"Num GPUs: {len(tf.config.list_physical_devices('GPU'))}")


model = CycleGAN('bce', n_resblocks=6)
poolA, poolB = ImagePool(50), ImagePool(50)

dataset = tfds.load('monet',batch_size=BATCH_SIZE,shuffle_files=True)
setA, setB = dataset['monet'], dataset['photo']

logging.info('Starting Training ...')
opt = tf.optimizers.Adam(0.002)
opt.build(model.get_disc_trainable_variables() + model.get_gen_trainable_variables())
for dataA, dataB in zip(setA, setB):
    imgA = dataA['image']
    imgB = dataB['image']

    imgA = poolA.query(imgA)
    imgB = poolB.query(imgB)
    
    with tf.GradientTape() as disc_tape, tf.GradientTape() as gen_tape:
        realA, realAscore, fakeB, fakeBscore, realA_regen = model.forward_A(imgA)
        realB, realBscore, fakeA, fakeAscore, realB_regen = model.forward_B(imgB)

        gan_loss = -model.gan_loss(realAscore, fakeAscore, realBscore, fakeBscore) # Negate as discriminator tries to maximise this value
        loss = model.cycle_loss(realA, realA_regen, realB, realB_regen) - gan_loss

    disc_grad = disc_tape.gradient(gan_loss, model.get_disc_trainable_variables())
    gen_grad = gen_tape.gradient(loss, model.get_gen_trainable_variables())

    opt.apply_gradients(zip(disc_grad, model.get_disc_trainable_variables()))
    opt.apply_gradients(zip(gen_grad, model.get_gen_trainable_variables()))

    logging.info(loss.numpy())




