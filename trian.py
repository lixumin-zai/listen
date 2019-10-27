from tensorflow.keras import layers
import tensorflow as tf
import numpy as np
import os
import get_more_data
from music21 import *

def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(1*25*64, use_bias=False, input_shape=(100,)))


    model.add(layers.Reshape((1, 25, 64)))
    assert model.output_shape == (None, 1, 25, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(32, (1, 5), strides=(1,2), padding='same', use_bias=False))
    assert model.output_shape == (None, 1, 50, 32)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    #
    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(4,2), padding='same', use_bias=False))
    assert model.output_shape == (None, 4, 100, 1)
    # model.add(layers.BatchNormalization())
    # model.add(layers.LeakyReLU())
    # # #
    # model.add(layers.Conv2DTranspose(1, (1, 5), strides=(1, 2), padding='same', use_bias=False, activation='tanh'))
    # assert model.output_shape == (None, 4, 100, 1)
    # # #
    return model

generator = make_generator_model()


# noise = tf.random.normal([1,100])
# generated_image = generator(noise, training=False)
# print(generated_image)
# input()


def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(32, (5,5), strides=(1,2), padding='same',input_shape=[4,100,1]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.1))

    model.add(layers.Conv2D(64, (5,5), strides=(1,2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.1))


    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model
# a = get_data.get_data()
# # print(a0)
# a0 = a[:, 0:200]
# a0 = a0.astype(np.float32)
# print(a0)
# b = tf.constant(a0)
# b = tf.reshape(b,(1,4,200,1))
discriminator = make_discriminator_model()
# decision = discriminator(b)
# print (decision)
# input()


cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)




# @tf.function
def train_step(real_output,generator,discriminator):
    noise = tf.random.normal((1, 100))

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      generated_images = generator(noise,training=True)

      real_output = discriminator(real_output,training=True)
      fake_output = discriminator(generated_images,training=True)

      gen_loss = generator_loss(fake_output)
      disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))


    return generated_images, gen_loss, disc_loss

# generator = make_generator_model()
# discriminator = make_discriminator_model()


checkpoint_dir = './training_checkpoints_trian'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

def train(b, es=100000):
    seed = tf.random.normal([1, 100])


    for e in range(es):
        output,gen_loss,disc_loss = train_step(b,generator,discriminator)
        if e % 1000 == 0:
            print(gen_loss,disc_loss)
            checkpoint.save(file_prefix=checkpoint_prefix)
        if e % 1000 == 0 and e != 0:

            predictions = generator(seed, training=False)
            print("第n次%d:"%e,predictions.numpy())
            pre = np.reshape(predictions.numpy(),[4,100])
            pre0 = np.trunc(np.abs(pre[0]))
            pre1 = np.trunc(np.abs(pre[1]))
            pre2 = np.around(np.abs(pre[2]),decimals =  2)
            pre3 = np.around(np.abs(pre[3]),decimals =  2)
            # print(pre)0
            # input()
            note0 = pre0.tolist()
            velocity0 = pre1.tolist()
            duration0 = pre2.tolist()
            time0 = pre3.tolist()
            print(note0,velocity0,duration0,time0)

            big = stream.Stream()
            temp = 0
            TimeSignature0 = None


            for el in range(0, len(time0)):
                # temp = temp + 1
                if temp < len(time0) - 10:
                    count = time0[temp: temp + 10].count(time0[temp])
                else:
                    count = time0[temp: len(time0)].count(time0[temp])
                if temp != len(time0) - 1:
                    if count == 1:
                        note_geti = note.Note(note0[temp])
                        note_geti.duration = duration.Duration(duration0[temp])
                        note_geti.volume.velocity = velocity0[temp]
                        big.insert(time0[temp], note_geti)
                        # note_geti.offset = time0[temp]
                        # measure.insert(time[temp],note_geti)
                        # big.append(note_geti)
                        temp = temp + 1
                    if count != 1 and temp + count < len(time0):
                        chord_geti = chord.Chord()
                        for el in range(0, count):
                            note_on_chord = note.Note(note0[temp + el])
                            note_on_chord.volume.velocity = velocity0[temp + el]
                            chord_geti.add(note_on_chord)
                        chord_geti.duration = duration.Duration(duration0[temp])
                        big.insert(time0[temp], chord_geti)

                        # chord_geti.offset = time0[temp]
                        # print(chord_geti.offset)
                        # measure.insert(time[temp], chord_geti)
                        temp = temp + count

            # for el in range(0,len(tempo_time)):
            #     tp =tempo.MetronomeMark(number=tempo0[el])
            #     if el < len(tempo_time):
            #         big.insert(tempo_time[el], tp)

            # ts = meter.TimeSignature('{}'.format(TimeSignature0))

            # big.insert(0, 3/4)
            big.show('midi')
            input()

# a0 = np.random.randint(0,128,(4,100))
a = get_more_data.get_more_data()
print(a)
# input()
# a0 = a[:, 0:100]
# # a0 = a0.T
# # a1 = a[:, 100:200]
# # print(a0.shape)
# # input()
# a0 = a0.astype(np.float32)
# # a1 = a1.astype(np.float32)
# b = tf.constant(a0)

b = tf.reshape(a,(29,4,100,1))
# print(b)
train(b)

# print()