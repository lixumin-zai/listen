from tensorflow.keras import layers
import tensorflow as tf
import numpy as np
import os
from music21 import *
import model1
import model2
import generate_music


model_name = input("choose modle:")
if model_name == 1:
    model = model1
else:
    model = model2

checkpoint_dir = './training_checkpoints_model{}'.format(model_name)
generator = model.make_generator_model()
discriminator = model.make_discriminator_model()
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=model.generator_optimizer,
                                 discriminator_optimizer=model.discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

def train(real_input, es=100000):
    for e in range(es):
        output,gen_loss,disc_loss = model.train_step(real_input,generator,discriminator)
        if e % 1000 == 0:
            print(gen_loss,disc_loss)
            checkpoint.save(file_prefix=checkpoint_prefix)

        if e % 10000 == 0 and e != 0:
            for i in range(0,10):
                generate_music.generate(generator,10)

def main():

    train()

if __name__ == '__main__':
    main()