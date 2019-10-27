from tensorflow.keras import layers
import tensorflow as tf
import numpy as np

def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(4*25*128, use_bias=False, input_shape=(100,)))


    model.add(layers.Reshape((4, 25, 128)))
    assert model.output_shape == (None, 4, 25, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (4, 5), strides=(1,2), padding='same', use_bias=False))
    assert model.output_shape == (None, 4, 50, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    #
    model.add(layers.Conv2DTranspose(1, (4, 5), strides=(1,2), padding='same', use_bias=False))
    model.add(layers.LeakyReLU())
    assert model.output_shape == (None, 4, 100, 1)
    # model.add(layers.BatchNormalization())
    # # # #
    # model.add(layers.Conv2DTranspose(16, (2, 1), strides=(2, 1), padding='valid', use_bias=False))
    # assert model.output_shape == (None, 2, 100, 16)
    # model.add(layers.BatchNormalization())
    # model.add(layers.LeakyReLU())
    #
    # model.add(layers.Conv2DTranspose(1, (2, 1), strides=(2, 1), padding='valid', use_bias=False, activation='tanh'))
    # assert model.output_shape == (None, 4, 100, 1)
    # # #
    return model



def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (4,1), strides=(1,1), padding='valid',input_shape=[4,100,1]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.1))

    model.add(layers.Conv2D(128, (1,5), strides=(1,2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.1))


    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model


cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

def generator_optimizer():
    return tf.keras.optimizers.Adam(1e-4)

def discriminator_optimizer():
    return tf.keras.optimizers.Adam(1e-4)


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

    generator_optimizer().apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer().apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))


    return generated_images, gen_loss, disc_loss





