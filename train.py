from ProGANVanilla import *
from Perceptual_loss_VGG import PROG_PL_VGG19
from util import *

import tensorflow as tf
import tensorflow_datasets as tfds
import os
from datetime import datetime
import matplotlib.pyplot as plt
"""
Training Progressive GAN, 
"""
### HYPERPARAMS ###
batch_size = 16
epochs = 256

# image #
UP_SAMPLE = 2 # factor for upsample
START_INPUT_DIM = 8 # start with 8x8 input -> initialize with growth phase to 4x4 (so really 4)
TARGET_DIM = 256 # full image size

# Adam #
lr=0.001
beta_1=0
beta_2=0.99
epsilon=10e-8

### MODELS ###
ProGAN = ProGAN()

# use 1st conv block for content loss
# input shape is the size of intial output of gan
# we intialize GAN as 2x2 input -> 4x4 output
# prior to training grow GAN to 4x4 -> 8x8 output
# this will need to be adjusted when looking at 4x and 8x
# it is important to note that VGG will not actually be used
# until the input size grows to 32x32
vgg = PROG_PL_VGG19(input_dims=(32, 32, 3),
                    layers_to_extract=[0, 1, 2],
                    load_weights='imagenet',
                    channel_last=True)

### DATA ###
"""
NWPU-RESISC45
This dataset requires you to download the source data manually 
into download_config.manual_dir (defaults to ~/tensorflow_datasets/manual/):

Note: this dataset does not have a test/train split.
"""
# load data #
data, info = tfds.load('resisc45', split="train", with_info=True)
# visualize data #
tfds.show_examples(data, info)

# size of entire dataset #
ds_size = info.splits["train"].num_examples
image_shape = info.features['image'].shape
# manually split ds into 80:20, train & test respectively #
test_ds_size = int(ds_size*0.20)
train_ds_size = ds_size - test_ds_size
# split #
test_ds = data.take(test_ds_size)
train_ds = data.skip(test_ds_size)
print("size of test: {}, size of train: {}".format(test_ds_size, train_ds_size))

# num features
num_features = info.features["label"].num_classes

# minibatch
test_ds = test_ds.batch(batch_size).repeat(epochs)
train_ds = train_ds.batch(batch_size).repeat(epochs)

# convert to np array
test_ds = tfds.as_numpy(test_ds)
train_ds = tfds.as_numpy(train_ds)


"""
A training batch will consist of generation of image for each sample,
train discrim on both generated images and real ones. 

1:2 ratio of samples 
"""

# update the alpha value on each instance of WeightedSum
def update_fadein(step):
    # calculate current alpha (linear from 0 to 1)
    # we only perform fadein in training #
    alpha = step / float(train_ds_size - 1)
    # update the alpha for each model
    ProGAN.set_alpha(alpha)
    
### loss functions ###
gen_loss = tf.keras.losses.BinaryCrossentropy()
discrim_loss = tf.keras.losses.BinaryCrossentropy()

### adam optimizer for SGD ###
optimizer = tf.keras.optimizers.Adam(lr=lr,
                                    beta_1=beta_1,
                                    beta_2=beta_2,
                                    epsilon=epsilon)

### intialize train metrics ###
gen_train_loss = tf.keras.metrics.Mean(name='gen_train_loss')
gen_train_accuracy = tf.keras.metrics.Accuracy(name='gen_train_accuracy')
dis_train_loss = tf.keras.metrics.Mean(name='dis_train_loss')
dis_train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='dis_train_accuracy')

### intialize test metrics ###
gen_test_loss = tf.keras.metrics.Mean(name='gen_test_loss')
gen_test_accuracy = tf.keras.metrics.Accuracy(name='gen_test_accuracy')
dis_test_loss = tf.keras.metrics.Mean(name='dis_test_loss')
dis_test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='dis_test_accuracy')

### tensorboard ###

# initialize logs #
current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
gen_train_log_dir = './logs/gradient_tape/' + current_time + '/gen_train'
gen_test_log_dir = './logs/gradient_tape/' + current_time + '/gen_test'
dis_train_log_dir = './logs/gradient_tape/' + current_time + '/dis_train'
dis_test_log_dir = './logs/gradient_tape/' + current_time + '/dis_test'

gen_train_summary_writer = tf.summary.create_file_writer(gen_train_log_dir)
gen_test_summary_writer = tf.summary.create_file_writer(gen_test_log_dir)
dis_train_summary_writer = tf.summary.create_file_writer(dis_train_log_dir)
dis_test_summary_writer = tf.summary.create_file_writer(dis_test_log_dir)
image_summary_writer = tf.summary.create_file_writer('./logs/gradient_tape/' + current_time + '/images')

### Weights Dir ###
if not os.path.isdir('./checkpoints'):
    os.mkdir('./checkpoints')

### generator train step ###
@tf.function
def gen_train_step(high_res, low_res, gan_output_res):
  with tf.GradientTape() as tape:
    # training=True is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    predictions = ProGAN.Generator(low_res, training=True)
    # mean squared error in prediction
    m_loss = tf.keras.losses.MSE(high_res, predictions)
    # we will only incorporate perceptual loss once we reach the dimensions of which the
    # vgg will accept. (32x32)
    if gan_output_res >= 32:
        # content loss
        v_pass = vgg(high_res)
        v_loss = tf.keras.losses.MSE(v_pass, predictions)
        # GAN loss + mse loss + feature loss
        loss = gen_loss(high_res, predictions) + v_loss + m_loss

    else:
        loss = gen_loss(high_res, predictions)  + m_loss

  # update either current or fadein
  if fadein:
      model = ProGAN.Generator._fadein_model
  else:
      model = ProGAN.Generator._current_model

  # apply gradients
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

  # update metrics
  gen_train_loss(loss)

  gen_train_accuracy(high_res, predictions)

  # write to gen_train-log #
  with gen_train_summary_writer.as_default():
      tf.summary.scalar('gen_train_loss', gen_train_loss.result(), step=epoch)
      tf.summary.scalar('gen_train_accuracy', gen_train_accuracy.result(), step=epoch)

  with image_summary_writer.as_default():
      tf.summary.image("Generated", predictions, step=0)
      tf.summary.image("high res", high_res, step=0)
      tf.summary.image("low res", low_res, step=0)


### discriminator train step ###
@tf.function
def dis_train_step(high_res, low_res, step):
    with tf.GradientTape() as tape:
        # discrim is a simple conv that performs binary classification
        # either SR or HR
        # use super res on even, true image on odd steps #
        label = step % 2
        # 0 for generated, 1 for true image
        if label:
            x = ProGAN.Generator(low_res, training=False)
        else:
            x = high_res
        # predict on gen output
        predictions = ProGAN.Discriminator(x, training=True)
        loss = discrim_loss(high_res, predictions)

    # update either current or fadein
    if fadein:
        model = ProGAN.Discriminator._fadein_model
    else:
        model = ProGAN.Discriminator._current_model

    # apply gradients
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    # update metrics
    dis_train_loss(loss)
    dis_train_accuracy(label, predictions)

    # write to dis_train-log #
    with dis_train_summary_writer.as_default():
        tf.summary.scalar('dis_train_loss', dis_train_loss.result(), step=epoch)
        tf.summary.scalar('dis_train_accuracy', dis_train_accuracy.result(), step=epoch)

### generator test step ###
@tf.function
def gen_test_step(high_res, low_res):
  # feed test sample in
  predictions = ProGAN.Generator(low_res, training=False)
  t_loss = gen_loss(high_res, predictions)

  # update metrics
  gen_test_loss(t_loss)
  gen_test_accuracy(high_res, predictions)

  # write to gen_test-log #
  with gen_test_summary_writer.as_default():
      tf.summary.scalar('gen_test_loss', gen_test_loss.result(), step=epoch)
      tf.summary.scalar('gen_test_accuracy', gen_test_accuracy.result(), step=epoch)


### discriminator test step ###
@tf.function
def dis_test_step(high_res, low_res, step):
    # feed test sample in
    # use super res on even, true image on odd steps #
    label = step % 2
    if label:
        x = ProGAN.Generator(low_res, training=False)
    else:
        x = high_res
    # predict on gen output
    predictions = ProGAN.Discriminator(x, training=False)
    t_loss = discrim_loss(high_res, predictions)

    # update metrics
    dis_test_loss(t_loss)
    dis_test_accuracy(label, predictions)

    # write to gen_test-log #
    with dis_test_summary_writer.as_default():
        tf.summary.scalar('dis_test_loss', dis_test_loss.result(), step=epoch)
        tf.summary.scalar('dis_test_accuracy', dis_test_accuracy.result(), step=epoch)


### TRAIN ###
def train(epoch, save_c, gan_output_res):
    """
    train step
    :param epoch: int epoch
    :param fadein: bool True for fadein (training)
    :return:
    """
    # Reset the metrics at the start of the next epoch
    gen_train_loss.reset_states()
    gen_train_accuracy.reset_states()
    gen_test_loss.reset_states()
    gen_test_accuracy.reset_states()

    dis_train_loss.reset_states()
    dis_train_accuracy.reset_states()
    dis_test_loss.reset_states()
    dis_test_accuracy.reset_states()
    # alternating training pattern
    if not epoch % 2:
        ### train generator on even epochs ###
        # iterator split into batches
        # apply alpha in training #
        # data structured: dataset -> batch -> sample
        for i, batch in enumerate(train_ds):
            for j, sample in enumerate(batch['image'].astype('int32')):


                high_res, low_res = preprocess(sample, (input_dim, input_dim), UP_SAMPLE)
                gen_train_step(high_res, low_res, gan_output_res)
                with image_summary_writer.as_default():
                    tf.summary.image("Ground Truth", sample[np.newaxis, ...], step=0)

        for batch in test_ds:
            for sample in batch:
                test_high_res, test_low_res = preprocess(sample['image'], (input_dim, input_dim), UP_SAMPLE)
                gen_test_step(test_high_res, test_low_res)

    else:
        ### train discriminator on odd epochs ###
        # data structured: dataset -> batch -> sample
        for i, batch in enumerate(train_ds):
            for j, sample in enumerate(batch['image'].astype('int32')):

                high_res, low_res = preprocess(sample['image'], (input_dim, input_dim), UP_SAMPLE)
                dis_train_step(high_res, low_res, i)


        for i, batch in enumerate(test_ds):
            for j, sample in enumerate(batch['image'].astype('int32')):
                test_high_res, test_low_res = preprocess(sample, (input_dim, input_dim), UP_SAMPLE)
                dis_test_step(test_high_res, test_low_res, i)


    ### save weights ###
    if not epoch % NUM_CHECKPOINTS_DIV:
        vgg.save_weights('./checkpoints/{}/{}'.format(current_time, save_c))
        save_c += 1


# initialize input_dim
input_dim = START_INPUT_DIM


# save count for checkpoints #
save_c = 0
NUM_CHECKPOINTS_DIV = int(epochs/4)

### TRAIN LOOP ###
"""
Two phases:
    1. Fade in the 3-layer block
    2. stabilize the network
"""
while input_dim <= TARGET_DIM:
    # train on given input dim #
    for epoch in range(epochs):

        # fadein #
        train(epoch, save_c=save_c, gan_output_res=input_dim)

        # stabilize#
        train(epoch, save_c=save_c, gan_output_res=input_dim)

    # grow input #
    ProGAN.grow() # upsamples by factor of 2

    # once vgg is used, grow! #
    if input_dim >= 32:
        vgg.grow()
    # reset alpha back to zero
    ProGAN.set_alpha(0.0)

    # increase input by upsample factor (2)
    input_dim*=UP_SAMPLE

"""
@article{Cheng_2017,
   title={Remote Sensing Image Scene Classification: Benchmark and State of the Art},
   volume={105},
   ISSN={1558-2256},
   url={http://dx.doi.org/10.1109/JPROC.2017.2675998},
   DOI={10.1109/jproc.2017.2675998},
   number={10},
   journal={Proceedings of the IEEE},
   publisher={Institute of Electrical and Electronics Engineers (IEEE)},
   author={Cheng, Gong and Han, Junwei and Lu, Xiaoqiang},
   year={2017},
   month={Oct},
   pages={1865-1883}
}
"""