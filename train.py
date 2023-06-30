import tensorflow as tf
from discriminator import Discriminator
from generator import Generator

discA, discB = Discriminator(), Discriminator()
genA, genB = Generator(n_resblocks=6), Generator(n_resblocks=6)



