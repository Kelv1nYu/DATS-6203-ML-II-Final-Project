# import
import glob
import os
import PIL
import imageio
import shutil
import numpy as np
from PIL import ImageDraw
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa
from tensorflow.keras.callbacks import Callback

AUTOTUNE = tf.data.experimental.AUTOTUNE
strategy = tf.distribute.get_strategy()

# Load data
MONET_FILENAMES = tf.io.gfile.glob(os.getcwd() + '/monet_tfrec/*.tfrec')
PHOTO_FILENAMES = tf.io.gfile.glob(os.getcwd() + '/photo_tfrec/*.tfrec')

# Data augment
def data_augment(image):
    p_spatial = tf.random.uniform([], 0, 1.0, dtype=tf.float32)
    p_rotate = tf.random.uniform([], 0, 1.0, dtype=tf.float32)

    # rotation
    if p_rotate > .8:
        image = tf.image.rot90(image)
    elif p_rotate > .6:
        image = tf.image.rot90(image, 2)
    elif p_rotate > .4:
        image = tf.image.rot90(image, 3)

    # flip
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    if p_spatial > .75:
        image = tf.image.transpose(image)

    # crop
    # image = tf.image.random_crop(image, size=[128, 128, 3])

    return image

# Normalize
def normalize_img(image):
    img = (tf.cast(image, dtype=tf.float32) / 127.5) - 1.
    return img

# Decode img
def decode_img(image):
    image = tf.image.decode_jpeg(image, channels=3)
    img = tf.reshape(image, [256, 256, 3])
    return img

# Record data
def read_tfrecord(example):
    tfrecord_format = {
        'image': tf.io.FixedLenFeature([], tf.string)
    }
    example = tf.io.parse_single_example(example, tfrecord_format)
    image = decode_img(example['image'])
    return image

# Load dataset
def load_dataset(filename):
    dataset = tf.data.TFRecordDataset(filename)
    dataset = dataset.map(read_tfrecord, num_parallel_calls=AUTOTUNE)
    return dataset

# Get dataset
def get_dataset(filenames, augment=None, repeat=True, batch_size=1):
    dataset = load_dataset(filenames)

    if repeat:
        dataset = dataset.repeat(2)

    if augment:
        dataset = dataset.map(augment, num_parallel_calls=AUTOTUNE)
    dataset = dataset.map(normalize_img, num_parallel_calls=AUTOTUNE)

    dataset = dataset.batch(batch_size)
    dataset = dataset.cache()
    dataset = dataset.prefetch(AUTOTUNE)

    return dataset


# Model functions
def encoder_block(filters, size=3, strides=2, apply_instancenorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)
    gamma_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)

    block = keras.Sequential()
    # Convolution
    block.add(layers.Conv2D(filters, size, strides, padding='same', kernel_initializer=initializer, use_bias=False))

    # Normalization
    if apply_instancenorm:
        block.add(tfa.layers.InstanceNormalization(gamma_initializer=gamma_init))

    # Activation
    block.add(layers.ReLU())

    return block


def decoder_block(filters, size=3, strides=2, apply_instancenorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)
    gamma_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)

    block = keras.Sequential()
    # Transposed convolutional layer
    block.add(
        layers.Conv2DTranspose(filters, size, strides, padding='same', kernel_initializer=initializer, use_bias=False))

    # Normalization
    if apply_instancenorm:
        block.add(tfa.layers.InstanceNormalization(gamma_initializer=gamma_init))

    # Activation
    block.add(layers.LeakyReLU())
    return block


def resnet_block(input_layer, size=3, strides=1):
    initializer = tf.random_normal_initializer(0., 0.02)
    gamma_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
    filters = input_layer.shape[-1]

    block = layers.Conv2D(filters, size, strides=strides, padding='same', use_bias=False,
                          kernel_initializer=initializer)(input_layer)
    block = tfa.layers.InstanceNormalization(gamma_initializer=gamma_init)(block)
    block = layers.ReLU()(block)

    block = layers.Conv2D(filters, size, strides=strides, padding='same', use_bias=False,
                          kernel_initializer=initializer)(block)
    block = tfa.layers.InstanceNormalization(gamma_initializer=gamma_init)(block)

    block = layers.Add()([block, input_layer])

    return block

# Generator
OUTPUT_CHANNELS = 3


def Generator():
    initializer = tf.random_normal_initializer(0., 0.02)
    gamma_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
    inputs = layers.Input(shape=[256, 256, 3])
    x = inputs

    down_layers = [encoder_block(64, 7, 1, apply_instancenorm=False),
                   encoder_block(128, 3, 2, apply_instancenorm=True),
                   encoder_block(256, 3, 2, apply_instancenorm=True)]

    skips = []
    for layer in down_layers:
        x = layer(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    for n in range(5):
        x = resnet_block(x, 3, 1)

    up_layers = [decoder_block(256, 3, 2),
                 decoder_block(128, 3, 2),
                 decoder_block(64, 3, 2), ]

    for layer, skip in zip(up_layers, skips):
        x = layer(x)
        x = layers.Concatenate()([x, skip])

    last = layers.Conv2D(OUTPUT_CHANNELS, 7,
                         strides=1, padding='same',
                         kernel_initializer=initializer,
                         use_bias=False,
                         activation='tanh')

    outputs = last(x)

    return keras.Model(inputs=inputs, outputs=outputs)

# discriminator
def Discriminator():
    initializer = tf.random_normal_initializer(0., 0.02)
    inputs = layers.Input(shape=[256, 256, 3])
    x = inputs
    x = encoder_block(64, 4, 2, apply_instancenorm=False)(x)  # (bs, 128, 128, 64)
    x = encoder_block(128, 4, 2, apply_instancenorm=True)(x)  # (bs, 64, 64, 128)
    x = encoder_block(256, 4, 2, apply_instancenorm=True)(x)  # (bs, 32, 32, 256)
    x = encoder_block(512, 4, 1, apply_instancenorm=True)(x)

    outputs = layers.Conv2D(1, 4, strides=1, padding='valid', kernel_initializer=initializer)(x)

    return keras.Model(inputs=inputs, outputs=outputs)

# Build model
with strategy.scope():
    monet_generator = Generator()  # transforms photos to Monet-esque paintings
    photo_generator = Generator()  # transforms Monet paintings to be more like photos

    monet_discriminator = Discriminator()  # differentiates real Monet paintings and generated Monet paintings
    photo_discriminator = Discriminator()  # differentiates real photos and generated photos


class CycleGan(keras.Model):
    def __init__(
            self,
            monet_generator,
            photo_generator,
            monet_discriminator,
            photo_discriminator,
            lambda_cycle=10,
    ):
        super(CycleGan, self).__init__()
        self.m_gen = monet_generator
        self.p_gen = photo_generator
        self.m_disc = monet_discriminator
        self.p_disc = photo_discriminator
        self.lambda_cycle = lambda_cycle

    def compile(
            self,
            m_gen_optimizer,
            p_gen_optimizer,
            m_disc_optimizer,
            p_disc_optimizer,
            gen_loss_fn,
            disc_loss_fn,
            cycle_loss_fn,
            identity_loss_fn
    ):
        super(CycleGan, self).compile()
        self.m_gen_optimizer = m_gen_optimizer
        self.p_gen_optimizer = p_gen_optimizer
        self.m_disc_optimizer = m_disc_optimizer
        self.p_disc_optimizer = p_disc_optimizer
        self.gen_loss_fn = gen_loss_fn
        self.disc_loss_fn = disc_loss_fn
        self.cycle_loss_fn = cycle_loss_fn
        self.identity_loss_fn = identity_loss_fn

    def train_step(self, batch_data):
        real_monet, real_photo = batch_data

        with tf.GradientTape(persistent=True) as tape:
            # photo to monet back to photo
            fake_monet = self.m_gen(real_photo, training=True)
            cycled_photo = self.p_gen(fake_monet, training=True)

            # monet to photo back to monet
            fake_photo = self.p_gen(real_monet, training=True)
            cycled_monet = self.m_gen(fake_photo, training=True)

            # generating itself
            same_monet = self.m_gen(real_monet, training=True)
            same_photo = self.p_gen(real_photo, training=True)

            # discriminator used to check, inputing real images
            disc_real_monet = self.m_disc(real_monet, training=True)
            disc_real_photo = self.p_disc(real_photo, training=True)

            # discriminator used to check, inputing fake images
            disc_fake_monet = self.m_disc(fake_monet, training=True)
            disc_fake_photo = self.p_disc(fake_photo, training=True)

            # evaluates generator loss
            monet_gen_loss = self.gen_loss_fn(disc_fake_monet)
            photo_gen_loss = self.gen_loss_fn(disc_fake_photo)

            # evaluates total cycle consistency loss
            total_cycle_loss = self.cycle_loss_fn(real_monet, cycled_monet, self.lambda_cycle) + self.cycle_loss_fn(
                real_photo, cycled_photo, self.lambda_cycle)

            # evaluates total generator loss
            total_monet_gen_loss = monet_gen_loss + total_cycle_loss + self.identity_loss_fn(real_monet, same_monet,
                                                                                             self.lambda_cycle)
            total_photo_gen_loss = photo_gen_loss + total_cycle_loss + self.identity_loss_fn(real_photo, same_photo,
                                                                                             self.lambda_cycle)

            # evaluates discriminator loss
            monet_disc_loss = self.disc_loss_fn(disc_real_monet, disc_fake_monet)
            photo_disc_loss = self.disc_loss_fn(disc_real_photo, disc_fake_photo)

        # Calculate the gradients for generator and discriminator
        monet_generator_gradients = tape.gradient(total_monet_gen_loss,
                                                  self.m_gen.trainable_variables)
        photo_generator_gradients = tape.gradient(total_photo_gen_loss,
                                                  self.p_gen.trainable_variables)

        monet_discriminator_gradients = tape.gradient(monet_disc_loss,
                                                      self.m_disc.trainable_variables)
        photo_discriminator_gradients = tape.gradient(photo_disc_loss,
                                                      self.p_disc.trainable_variables)

        # Apply the gradients to the optimizer
        self.m_gen_optimizer.apply_gradients(zip(monet_generator_gradients,
                                                 self.m_gen.trainable_variables))

        self.p_gen_optimizer.apply_gradients(zip(photo_generator_gradients,
                                                 self.p_gen.trainable_variables))

        self.m_disc_optimizer.apply_gradients(zip(monet_discriminator_gradients,
                                                  self.m_disc.trainable_variables))

        self.p_disc_optimizer.apply_gradients(zip(photo_discriminator_gradients,
                                                  self.p_disc.trainable_variables))

        return {
            "monet_gen_loss": total_monet_gen_loss,
            "photo_gen_loss": total_photo_gen_loss,
            "monet_disc_loss": monet_disc_loss,
            "photo_disc_loss": photo_disc_loss
        }

# Loss Function
with strategy.scope():
    def discriminator_loss(real, generated):
        real_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)(tf.ones_like(real), real)

        generated_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)(tf.zeros_like(generated), generated)

        total_disc_loss = real_loss + generated_loss

        return total_disc_loss * 0.5

    def generator_loss(generated):
        return tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)(tf.ones_like(generated), generated)

    def calc_cycle_loss(real_image, cycled_image, LAMBDA):
        loss1 = tf.reduce_mean(tf.abs(real_image - cycled_image))

        return LAMBDA * loss1

    def identity_loss(real_image, same_image, LAMBDA):
        loss = tf.reduce_mean(tf.abs(real_image - same_image))
        return LAMBDA * 0.5 * loss

# #optimizer
with strategy.scope():
    monet_generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    photo_generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

    monet_discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    photo_discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

#compile
with strategy.scope():
    cycle_gan_model = CycleGan(
        monet_generator, photo_generator, monet_discriminator, photo_discriminator
    )

    cycle_gan_model.compile(
        m_gen_optimizer = monet_generator_optimizer,
        p_gen_optimizer = photo_generator_optimizer,
        m_disc_optimizer = monet_discriminator_optimizer,
        p_disc_optimizer = photo_discriminator_optimizer,
        gen_loss_fn = generator_loss,
        disc_loss_fn = discriminator_loss,
        cycle_loss_fn = calc_cycle_loss,
        identity_loss_fn = identity_loss
    )

# Monitor
class GANMonitor(Callback):

    def __init__(self, num_img=1, monet_path='monet', photo_path='photo'):
        self.num_img = num_img
        self.monet_path = monet_path
        self.photo_path = photo_path
        # Create directories to save the generate images
        if not os.path.exists(self.monet_path):
            os.makedirs(self.monet_path)
        if not os.path.exists(self.photo_path):
            os.makedirs(self.photo_path)

    def on_epoch_end(self, epoch, logs=None):
        # Monet generated images
        for i, img in enumerate(photo_eval.take(self.num_img)):
            prediction = monet_generator(img, training=False)[0].numpy()
            prediction = (prediction * 127.5 + 127.5).astype(np.uint8)
            prediction = PIL.Image.fromarray(prediction)
            prediction.save(f'{self.monet_path}/generated_{i}_{epoch + 1}.png')

        # Photo generated images
        for i, img in enumerate(monet_eval.take(self.num_img)):
            prediction = photo_generator(img, training=False)[0].numpy()
            prediction = (prediction * 127.5 + 127.5).astype(np.uint8)
            prediction = PIL.Image.fromarray(prediction)
            prediction.save(f'{self.photo_path}/generated_{i}_{epoch + 1}.png')

monet_ds = get_dataset(MONET_FILENAMES, augment=data_augment)
photo_ds = get_dataset(PHOTO_FILENAMES, repeat=False)

monet_eval = get_dataset(MONET_FILENAMES, repeat=False)
photo_eval = get_dataset(PHOTO_FILENAMES, repeat=False)

# create gif
def create_gif(images_path, gif_path):
    images = []
    filenames = glob.glob(images_path)
    filenames.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))
    for epoch, filename in enumerate(filenames):
        img = PIL.ImageDraw.Image.open(filename)
        ImageDraw.Draw(img).text((0, 0),  # Coordinates
                                 f'Epoch {epoch + 1}')
        images.append(img)
    imageio.mimsave(gif_path, images, fps=2)


# train
cycle_gan_model.fit(tf.data.Dataset.zip((monet_ds, photo_ds)), epochs=10, callbacks=[GANMonitor()])

# create gif
create_gif(os.getcwd() + '/monet/*.png', 'monet.gif')
create_gif(os.getcwd() + '/photo/*.png', 'photo.gif')

print('creat complete')
