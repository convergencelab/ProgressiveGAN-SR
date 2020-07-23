from ProGANVanilla import *
from Perceptual_loss_VGG import PROG_PL_VGG19
from util import *
from Layers import *
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow.keras.backend as K
from ann_visualizer.visualize import ann_viz

import os
from datetime import datetime
import functools

"""
Training Progressive GAN, 

Should be training critic several times more than generator,
typical ratio is 5 critic updates:1 generator update
"""
### HYPERPARAMS ###
batch_size = 16
epochs = 256 # double for actually num iterations, as one epoch for fadein and one for straight pass
dis_per_gen_ratio = 5# number of critic trains per gen train
LAMBDA = 10# lambda for gradient penalty
# image #
UP_SAMPLE = 2 # factor for upsample
START_INPUT_DIM = 16 # start with 4x4 input -> initialize with growth phase to 8x8 (so really 4)
TARGET_DIM = 256 # full image size

# Adam #
# generator tends to take 4x less to train therefore two different learning rates:
gen_lr=0.01
dis_lr=0.01
beta_1=0.5
beta_2=0.9
epsilon=10e-8
# clip_constraint = 0.01 # clip used for w loss

### MODELS ###
# ProGAN = ProGAN(clip_constraint)
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

def preprocess(img_dict, lr_dim, upscale_factor=UP_SAMPLE):
    """
    preprocess an image to be lr hr pair for a given dim
    :param img: full size img
    :param lr_dim: dims for low res
    :param upscale_factor: upscale factor for hr
    :return: lr hr pair
    """
    hr_dim = tuple([i * upscale_factor for i in lr_dim])
    img = img_dict['image']
    # resize and normalize
    img_dict['image'] = tf.image.resize(img, hr_dim)/255.0, tf.image.resize(img, lr_dim)/255.0


    return img_dict

def prepare_and_upscale(lr_dim):
    """
    take base train and test and return the upscaled version
    this is all in one function so we dont have to keep
    the full size images in memory the entire training time

    lr_dim: the lower dim for the training step
    """
    # load data #
    data, info = tfds.load('resisc45', split="train", with_info=True)
    # num features
    num_features = info.features["label"].num_classes

    # visualize data #
    # tfds.show_examples(data, info)

    # manually split ds into 80:20, train & test respectively #
    ds_size = info.splits["train"].num_examples
    test_ds_size = int(ds_size * 0.20)
    train_ds_size = ds_size - test_ds_size

    # split #
    test_ds = data.take(test_ds_size)
    train_ds = data.skip(test_ds_size)

    # minibatch
    test_ds = test_ds.batch(batch_size)
    train_ds = train_ds.batch(batch_size)

    # preprocess mapping function, takes image and turns into
    # tuple with lr and hr img
    pp = functools.partial(preprocess, lr_dim=lr_dim)
    test_ds = test_ds.map(pp)
    train_ds = train_ds.map(pp)

    # convert to numpy #
    test_ds = tfds.as_numpy(test_ds)
    train_ds = tfds.as_numpy(train_ds)

    return train_ds, test_ds, train_ds_size


"""
Above we will see that the preprocessed datasets will have follwing features:
    generator
    -> each iter of generator contains an dict
        -> dict['images'] contains the images for the sample
            -> each dict['images'] contains a tuple with the lr and hr images 
            with specified batch size (dtype = np array)
"""

# update the alpha value on each instance of WeightedSum
def update_fadein(step, train_ds_size):
    # calculate current alpha (linear from 0 to 1)
    # we only perform fadein in training #
    alpha = step / float(train_ds_size - 1)
    # update the alpha for each model
    ProGAN.set_alpha(alpha)

### loss functions ###
gen_loss = tf.keras.losses.BinaryCrossentropy()
discrim_loss = tf.keras.losses.BinaryCrossentropy()

### adam optimizer for SGD ###
gen_optimizer = tf.keras.optimizers.Adam(lr=gen_lr,
                                     beta_1=beta_1,
                                     beta_2=beta_2,
                                     epsilon=epsilon)

dis_optimizer = tf.keras.optimizers.Adam(lr=dis_lr,
                                     beta_1=beta_1,
                                     beta_2=beta_2,
                                     epsilon=epsilon)

# This method returns a helper function to compute cross entropy loss
#cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

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
gen_train_log_dir = '../logs/gradient_tape/' + current_time + '/gen_train'
gen_test_log_dir = '../logs/gradient_tape/' + current_time + '/gen_test'
dis_train_log_dir = '../logs/gradient_tape/' + current_time + '/dis_train'
dis_test_log_dir = '../logs/gradient_tape/' + current_time + '/dis_test'
alpha_log_dir = '../logs/gradient_tape/' + current_time + '/alpha'
dis_graph_writer = tf.summary.create_file_writer('../logs/gradient_tape/' + current_time + '/dis_graph')
gen_graph_writer = tf.summary.create_file_writer('../logs/gradient_tape/' + current_time + '/gen_graph')

gen_train_summary_writer = tf.summary.create_file_writer(gen_train_log_dir)
gen_test_summary_writer = tf.summary.create_file_writer(gen_test_log_dir)
dis_train_summary_writer = tf.summary.create_file_writer(dis_train_log_dir)
dis_test_summary_writer = tf.summary.create_file_writer(dis_test_log_dir)
image_summary_writer = tf.summary.create_file_writer('../logs/gradient_tape/' + current_time + '/images')
alpha_summary_writer = tf.summary.create_file_writer(alpha_log_dir)

# start tensorboard
os.system("start /B start cmd.exe @cmd /c tensorboard --logdir={}".format('../logs/gradient_tape/' + current_time))

### Weights Dir ###
if not os.path.isdir('../checkpoints'):
    os.mkdir('../checkpoints')


# critic loss for discriminator
def discriminator_loss(real_output, fake_output):
    #### fake - real
    ### larger discrim = smaller gen loss, etc...
    # must also incorporate gradient penalty
    # measures the squared difference between the norm of
    # the gradient of the predictions with the respect to
    # input images and 1
    # intractable to evaluate every point, uses a handful of points
    # incorporate this into a layer

    interpolated_img = RandomWeightedAverage(batch_size)([real_output, fake_output])
    # interpolated image pass
    gradients = tf.gradients(ProGAN.Discriminator(interpolated_img), [interpolated_img])[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
    gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)


    return tf.reduce_mean(fake_output) - tf.reduce_mean(real_output) + LAMBDA * gradient_penalty


# discrim loss + vgg loss
def vgg_generator_loss(dis_output, high_res_imgs, predictions):
    # content loss
    v_pass = vgg(high_res_imgs)
    tf.print("shape of vpass: ", v_pass[-1].shape)
    v_pass_pred = vgg(predictions)
    tf.print("shape of vpass_pred: ", v_pass_pred[-1].shape)
    v_loss = tf.keras.losses.MSE(v_pass[-1], v_pass_pred[-1])
    # GAN loss + mse loss + feature loss
    # return vgg loss + dicrim output
    return v_loss - tf.reduce_mean(dis_output)


def generator_loss(dis_output):
    return -tf.reduce_mean(dis_output)


@tf.function
def dis_train_step(high_res_imgs, low_res_imgs, step):
    with tf.GradientTape() as dis_tape:
        generated_imgs = ProGAN.Generator(low_res_imgs, training=False)

        real_output = ProGAN.Discriminator(high_res_imgs, training=True)
        fake_output = ProGAN.Discriminator(generated_imgs, training=True)

        #gp = gradient_penalty(batch_size, high_res_imgs, generated_imgs)
        # dis loss...
        dis_loss = discriminator_loss(real_output, fake_output) #two passes : negative fake and regular real.

        # apply loss
        dis_train_loss(dis_loss)

        # apply accuracy
        dis_train_accuracy.update_state(real_output, fake_output)

        gradients_of_discriminator = dis_tape.gradient(dis_loss, ProGAN.Discriminator.trainable_variables)
        dis_optimizer.apply_gradients(zip(gradients_of_discriminator, ProGAN.Discriminator.trainable_variables))

    # write to dis_train-log #
    with dis_train_summary_writer.as_default():
        tf.summary.scalar('dis_train_loss', dis_train_loss.result(), step=step)
        tf.summary.scalar('dis_train_accuracy', dis_train_accuracy.result(), step=step)


@tf.function
def gen_train_step(high_res_imgs, low_res_imgs, step):
    with tf.GradientTape() as gen_tape:
        # pass thru gen
        generated_imgs = ProGAN.Generator(low_res_imgs, training=True)
        fake_output = ProGAN.Discriminator(generated_imgs, training=False)


        # if >= 32 use vgg loss!
        true_fn = lambda: vgg_generator_loss(fake_output, high_res_imgs, generated_imgs)
        false_fn = lambda: generator_loss(fake_output)
        gen_loss = tf.cond(input_dim >= 32, true_fn, false_fn)

        # apply loss
        gen_train_loss(gen_loss)

        # apply accuracy
        gen_train_accuracy.update_state(high_res_imgs, generated_imgs)

        gradients_of_generator = gen_tape.gradient(gen_loss, ProGAN.Generator.trainable_variables)
        gen_optimizer.apply_gradients(zip(gradients_of_generator, ProGAN.Generator.trainable_variables))
        # if epoch == 0 and step == 0:
           #  ann_viz(ProGAN.Generator, title="My first neural network")
    # write to gen_train-log #
    with gen_train_summary_writer.as_default():
        tf.summary.scalar('gen_train_loss', gen_train_loss.result(), step=step)
        tf.summary.scalar('gen_train_accuracy', gen_train_accuracy.result(), step=step)

    # view images #
    with image_summary_writer.as_default():
        tf.summary.image("Generated", generated_imgs, max_outputs=1, step=GROW_COUNT)
        tf.summary.image("high res", high_res_imgs, max_outputs=1, step=GROW_COUNT)
        tf.summary.image("low res", low_res_imgs, max_outputs=1, step=GROW_COUNT)


def train_epoch(epoch, save_c):
    # Reset the metrics at the start of the next epoch
    gen_train_loss.reset_states()
    #gen_train_accuracy.reset_states()
    gen_test_loss.reset_states()
    #gen_test_accuracy.reset_states()

    dis_train_loss.reset_states()
    dis_train_accuracy.reset_states()
    dis_test_loss.reset_states()
    dis_test_accuracy.reset_states()

    ## train gen ##
    for i, batch in enumerate(train_ds):
        step = tf.convert_to_tensor(i, dtype=tf.int64)
        # data structured: dataset -> batch -> sample
        hr, lr = batch['image']

        fk = ProGAN.Generator(lr).numpy()[0]
        tf.print("avg pixel: {}".format(np.average(fk, axis=0)))

        tf.summary.trace_on(graph=True, profiler=True)
        gen_train_step(hr, lr, step)
        # write graph
        with gen_graph_writer.as_default():
            tf.summary.trace_export(
                name="gen_graph_trace",
                step=0,
                profiler_outdir='../logs/gradient_tape/' + current_time + '/gen_graph')

    ## train discrim ##
    for i in range(dis_per_gen_ratio):
        for i, batch in enumerate(train_ds):
            step = tf.convert_to_tensor(i, dtype=tf.int64)
            # data structured: dataset -> batch -> sample
            hr, lr = batch['image']

            tf.summary.trace_on(graph=True, profiler=True)
            dis_train_step(hr, lr, step)
            # write graph
            with dis_graph_writer.as_default():
                tf.summary.trace_export(
                    name="dis_graph_trace",
                    step=0,
                    profiler_outdir='../logs/gradient_tape/' + current_time + '/dis_graph')

    # test #
    #for i, batch in enumerate(test_ds):
    #    step = tf.convert_to_tensor(i, dtype=tf.int64)
    #    # get lr, hr batch tuple
     #   hr, lr = batch['image']

    ### save weights ###
    if not epoch % NUM_CHECKPOINTS_DIV:
        vgg.save_weights('./checkpoints/{}/{}'.format(current_time, save_c))
        save_c += 1

# initialize input_dim
input_dim = tf.convert_to_tensor(START_INPUT_DIM, dtype=tf.int64)
# save count for checkpoints #
save_c = 0
NUM_CHECKPOINTS_DIV = int(epochs/4)

### TRAIN LOOP ###
"""
Two phases:
    1. Fade in the 3-layer block
    2. stabilize the network
"""
# intialize images at 8x8, 16x16
train_ds, test_ds, train_ds_size = prepare_and_upscale(lr_dim=(input_dim, input_dim))
GROW_COUNT = 0
while input_dim <= TARGET_DIM:
    # train on given input dim #
    if GROW_COUNT == 0:
        # set alpha to one for initial straight pass:
        ProGAN.set_alpha(1.0)
        for epoch in range(epochs):
            train_epoch(epoch, save_c=save_c)
    else:
        # all other growth phases fadein first.
        for epoch in range(epochs):
            # fadein #
            train_epoch(epoch, save_c=save_c)
            update_fadein(epoch, epochs)
            print("epoch {} for dims {} in fadein".format(epoch, (input_dim, input_dim)))
            with alpha_summary_writer.as_default():
                tf.summary.scalar('gen_train_loss', ProGAN.Generator.layers[-2].alpha, step=epoch)
        # stabalize for epoch #
        for epoch in range(epochs):
            # alpha=1.0 --> stabilize #
            train_epoch(epoch, save_c=save_c)
            print("epoch {} for dims {} in straightpass".format(epoch, (input_dim, input_dim)))

    # grow input #
    ProGAN.grow() # upsamples by factor of 2

    # once vgg is used, grow! #
    if input_dim >= 32:
        vgg.grow()

    # reset alpha back to zero
    ProGAN.set_alpha(0.0)

    # increase input by upsample factor (2)
    input_dim*=UP_SAMPLE
    # LOW_RES, HIGH_RES = (input_dim, input_dim), (input_dim*UP_SAMPLE, input_dim*UP_SAMPLE)
    # grow image sizes #
    train_ds, test_ds, train_ds_size = prepare_and_upscale(lr_dim=(input_dim.numpy(), input_dim.numpy()))
    GROW_COUNT += 1