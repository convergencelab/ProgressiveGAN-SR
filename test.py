"""
testing PL VGG-19 to have adjusted input sizes
"""

from ProGANVanilla import *
from Perceptual_loss_VGG import PROG_PL_VGG19
from util import *

import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np
"""
vgg = PROG_PL_VGG19(input_dims=(32, 32, 3),
                    layers_to_extract=[0, 1, 2],
                    load_weights='imagenet',
                    channel_last=True)


# load data #
data, info = tfds.load('resisc45', split="train", with_info=True)
# visualize data #
# tfds.show_examples(data, info)

# size of entire dataset #
ds_size = info.splits["train"].num_examples
image_shape = info.features['image'].shape

ds = tfds.as_numpy(data)

sample = next(ds)['image']

"""
### 32x32 ###
# samples = preprocess(img=sample, lr_dim=(16, 16), upscale_factor=2)
# sample = np.array(samples[1])[np.newaxis, ...]
# sample = tf.keras.applications.vgg19.preprocess_input(sample)
"""
# view output
out = vgg(sample)[2].numpy().transpose()
fig, ax = plt.subplots(5,5)
for l in range(5):
    for j in range(5):
        arr = np.reshape(out[l+j], (32, 32))
        ax[l, j].imshow(arr)

plt.show()
"""
"""
### 64x64 ###
sample = next(ds)['image']
samples = preprocess(img=sample, lr_dim=(32, 32), upscale_factor=2)

fig, ax = plt.subplots(1,3)
ax[0].imshow(samples[0])
ax[0].set_title("32x32")
ax[1].imshow(samples[1])
ax[1].set_title("64x64")
ax[2].imshow(sample)
ax[2].set_title("Ground Truth")

sample = np.array(samples[1])[np.newaxis, ...]
sample = tf.keras.applications.vgg19.preprocess_input(sample)

vgg.grow()
out = vgg(sample)[2].numpy().transpose()
fig, ax = plt.subplots(5,5)
for l in range(5):
    for j in range(5):
        arr = np.reshape(out[l+j], (64, 64))
        ax[l, j].imshow(arr)



"""

"""
fig, ax = plt.subplots(1,3)
ax[0].imshow(samples[0])
ax[0].set_title("16x16")
ax[1].imshow(samples[1])
ax[1].set_title("32x32")
ax[2].imshow(sample)
ax[2].set_title("Ground Truth")
"""
#plt.show()

### investigating data batches ###
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
print(image_shape)
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
test_ds = test_ds.batch(15).repeat(2)
train_ds = train_ds.batch(15).repeat(2)

# convert to np array
test_ds = tfds.as_numpy(test_ds)
train_ds = tfds.as_numpy(train_ds)

sample = next(train_ds)['image'][0]
hr, lr = preprocess(img=sample, lr_dim=(4, 4), upscale_factor=2)



### checking outputs of gan ###
"""
error:
    ValueError: Cannot reshape a tensor with 32768 elements to shape [1,4,4,128] (2048 elements) 
    for '{{node prog__generator/model_6/reshape/Reshape}} = Reshape[T=DT_FLOAT, Tshape=DT_INT32]
    (prog__generator/model_6/dense_1/BiasAdd, prog__generator/model_6/reshape/Reshape/shape)' with input shapes: 
    [1,4,4,2048], [4] and with input tensors computed as partial shapes: input[1] = [1,4,4,128].


    *** DISCRIMINATOR IS WORKING SO JUST NEED TO DEBUG GENERATOR ***
    got working after flatten: reshape was too small had to increase to 2048
"""
# TODO: figure out why generator is not taking input properly
ProGAN = ProGAN()
ProGAN.grow()
# dis is working!
# x = ProGAN.Discriminator(hr, training=True, fadein=True)


x = ProGAN.Generator(lr, training=True, fadein=True)
x = x.numpy().reshape(8,8,3)
fig, ax = plt.subplots(1,4)
ax[0].imshow(lr.reshape(4,4,3))
ax[1].imshow(x)
ax[2].imshow(hr.reshape(8,8,3))
ax[3].imshow(sample)

ax[0].set_title('low res')
ax[1].set_title('generated')
ax[2].set_title('high res')
ax[3].set_title('ground truth')

plt.show()