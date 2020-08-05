import tensorflow as tf
from tensorflow_examples.models.pix2pix import pix2pix
import tensorflow_datasets as tfds
from IPython.display import clear_output
import matplotlib.pyplot as plt
import functools
from datetime import datetime
import os
from PIL import Image

def preprocess(img_dict, lr_dim, upscale_factor=2):
    """
    preprocess an image to be lr hr pair for a given dim
    FORMAT (HR, LR)
    :param img: full size img
    :param lr_dim: dims for low res
    :param upscale_factor: upscale factor for hr
    :return: lr hr pair
    """
    hr_dim = tuple([i * upscale_factor for i in lr_dim])
    img = img_dict['image']
    # resize and normalize
    img_dict['image'] = tf.image.resize(img, lr_dim)/255.0, tf.image.resize(img, hr_dim)/255.0
    return img_dict['image']

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
    test_ds = test_ds.batch(BATCH_SIZE)
    train_ds = train_ds.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).cache().repeat()
    train_ds = train_ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    # preprocess mapping function, takes image and turns into
    # tuple with lr and hr img
    pp = functools.partial(preprocess, lr_dim=lr_dim)
    test_ds = test_ds.map(pp, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    train_ds = train_ds.map(pp, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # convert to numpy #
    test_ds = tfds.as_numpy(test_ds)
    train_ds = tfds.as_numpy(train_ds)

    return train_ds, test_ds, train_ds_size


# hyper params
BATCH_SIZE = 1
BUFFER_SIZE = 1000
MSE_WEIGHT = 0.4

# test 2x upscale
train, test, ds_size = prepare_and_upscale(lr_dim=(128, 128))
TRAIN_LENGTH = ds_size *0.80
STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE

def display(display_list, TB=False):
    if TB:
        return display_list
    plt.figure(figsize=(15, 15))
    title = ['Low Res', 'High Res', 'Generated']
    fig = plt.figure()
    for i in range(len(display_list)):
        sub = fig.add_subplot(1, len(display_list), i+1)
        sub.set_title(title[i])
        sub.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        sub.axis('off')

    plt.show()

# get sample
sample_lr, sample_hr = next(train)
# display
display([sample_lr[0], sample_hr[0]])

"""
Network being used is a modified unet-> unet consists of an encoder (downsampler)
and decoder (upsampler). In order to reduce amount of trainable params,
a pretrained model can be used as an encoder
for this task will use a mobilenetV2 -> intermediate outputs will be used
decoder will be the upsample block implemented in tensorflows pix2pix tutorial
"""
OUTPUT_CHANNELS = 3

base_model = tf.keras.applications.MobileNetV2(input_shape=[128, 128, 3], include_top=False)

# Use the activations of these layers
layer_names = [
    'block_1_expand_relu',   # 64x64
    'block_3_expand_relu',   # 32x32
    'block_6_expand_relu',   # 16x16
    'block_13_expand_relu',  # 8x8
    'block_16_project',      # 4x4
]

layers = [base_model.get_layer(name).output for name in layer_names]

# Create the feature extraction model
# not utilizing every layer in model hence this method of getting model.
down_stack = tf.keras.Model(inputs=base_model.input, outputs=layers)

# we want to use transfer learning -> essentially initialize with very
# good weights
down_stack.trainable = True

# upstack is created using upsampling layers from pix2pix example.
up_stack = [
    pix2pix.upsample(512, 3),  # 4x4 -> 8x8
    pix2pix.upsample(256, 3),  # 8x8 -> 16x16
    pix2pix.upsample(128, 3),  # 16x16 -> 32x32
    pix2pix.upsample(64, 3),   # 32x32 -> 64x64
    pix2pix.upsample(32, 3),   # 64x64 -> 128x128
    pix2pix.upsample(16, 3),   # 128x128 -> 256x256
]

class unet(tf.keras.Model):
    def __init__(self):
        super(unet, self).__init__(name='unet')
        # self.input = tf.keras.layers.Input(shape=[128, 128, 3])
        self.down_stack = down_stack
        self.up_stack = up_stack
        self.last = tf.keras.layers.Conv2DTranspose(
                                                3, 3, strides=2,
                                                padding='same')

        self.last2 = tf.keras.layers.Conv2DTranspose(
                                                    3, 3, strides=2,
                                                    padding='same')
    def call(self, inputs):
        skips = self.down_stack(inputs)
        x = skips[-1]
        skips = reversed(skips[:-1])

        # Upsampling and establishing the skip connections
        for up, skip in zip(self.up_stack, skips):
            x = up(x)
            concat = tf.keras.layers.Concatenate()
            x = concat([x, skip])

        x = self.last(x)
        x = self.last2(x)

        return x

def unet_model(output_channels=3):
  inputs = tf.keras.layers.Input(shape=[128, 128, 3])
  x = inputs

  # Downsampling through the model
  skips = down_stack(x)
  x = skips[-1]
  skips = reversed(skips[:-1])

  # Upsampling and establishing the skip connections
  for up, skip in zip(up_stack, skips):
    x = up(x)
    concat = tf.keras.layers.Concatenate()
    x = concat([x, skip])

  # This is the last layer of the model
  last = tf.keras.layers.Conv2DTranspose(
      output_channels, 3, strides=2,
      padding='same')
  # This is the last layer of the model
  last2 = tf.keras.layers.Conv2DTranspose(
      output_channels, 3, strides=2,
      padding='same')
  x = last2(last(x))

  return tf.keras.Model(inputs=inputs, outputs=x)


########
# Loss #
########
# vgg loss
vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
### feature layers ###
# model must extract featrues to classify, we can use this for
# feature extraction for non-classifcation purposes
# several layers for style

# use block 5 for content
content_layers = ['block1_conv1',
                  'block2_conv1',
                  'block3_conv1',
                  'block4_conv1',
                  'block5_conv1',
                  'block5_conv2']

def vgg_layers(layer_names):
  """ Creates a vgg model that returns a list of intermediate output values."""
  # Load our model. Load pretrained VGG, trained on imagenet data
  vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
  vgg.trainable = False
  # get each layer from vgg
  outputs = [vgg.get_layer(name).output for name in layer_names]
  # take vgg input + selected layers.
  model = tf.keras.Model([vgg.input], outputs)
  return model

vgg = vgg_layers(content_layers)

content_weight=1e4
num_content_layers = len(content_layers)
def content_style_MSE_loss(gen, hr):
  """
  Loss incorporates content loss to take actual perceptual differecnes into account
  MSE allows for the pixelwise difference -> MSE = L2 loss
  Will use MAE, less sensitive to outliers. -> MAE = L1 loss
  (SR-GAN uses L1)
  """
  hr_content = vgg(hr)
  gen_content = vgg(gen)

  # for i, j in zip(gen_content, hr_content):
   #      print(i.shape, j.shape)

  print(gen_content[-1].shape, hr_content[-1].shape)

  content_loss = tf.keras.losses.MSE(hr_content[-1], gen_content[-1])
  content_loss *= content_weight / num_content_layers

  # MAE = tf.keras.losses.MAE(hr, gen)
  # loss = content_loss + MAE

  return content_loss

###############
# Tensorboard #
###############
# initialize logs #
current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
train_summary_writer = tf.summary.create_file_writer('../logs/gradient_tape/' + current_time + '/train')
test_summary_writer = tf.summary.create_file_writer('../logs/gradient_tape/' + current_time + '/test')
image_summary_writer = tf.summary.create_file_writer('../logs/gradient_tape/' + current_time + '/images')

# start tensorboard #
os.system("start /B start cmd.exe @cmd /c tensorboard --logdir={}".format('../logs/gradient_tape/' + current_time))

### Weights Dir ###
if not os.path.isdir('../checkpoints'):
    os.mkdir('../checkpoints')

################
# TRAIN PARAMS #
################
EPOCHS = 20
VAL_SUBSPLITS = 5
VALIDATION_STEPS = (ds_size*0.20)//BATCH_SIZE//VAL_SUBSPLITS
lr=0.001
beta_1=0.5
beta_2=0.9
epsilon=10e-8

### adam optimizer for SGD ###
optimizer = tf.keras.optimizers.Adam(lr=lr,
                                     beta_1=beta_1,
                                     beta_2=beta_2,
                                     epsilon=epsilon)
# init model
model = unet_model()

def show_predictions(**kwargs):
    return display([sample_lr[0], sample_hr[0],
             # create mask based on model prediction
             tf.squeeze(model.predict(sample_lr[0][tf.newaxis, ...]))], **kwargs)

# view model output
show_predictions()
# metrics
epoch_loss_avg = tf.keras.metrics.Mean(name='loss')
epoch_accuracy = tf.keras.metrics.Accuracy(name='train_accuracy')

@tf.function
def train_step(x, y, epoch):
    with tf.GradientTape() as tape:
        generated = model(x)
        loss = content_style_MSE_loss(generated, y)
        grads = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

    # track results #
    epoch_accuracy.update_state(y, model(x))
    epoch_loss_avg.update_state(loss)

    # write to tensorboard #
    # write to dis_train-log #
    with train_summary_writer.as_default():
        tf.summary.scalar('train_loss', epoch_loss_avg.result(), step=epoch)
        tf.summary.scalar('train_accuracy', epoch_accuracy.result(), step=epoch)

    # view images #
    gen = model(x, training=False)
    with image_summary_writer.as_default():
        tf.summary.image("Generated", gen, max_outputs=3, step=epoch)
        tf.summary.image("high res", y, max_outputs=3, step=epoch)
        tf.summary.image("low res", x, max_outputs=3, step=epoch)

### TRAIN ###
save_c = 0
NUM_CHECKPOINTS_DIV = int(EPOCHS/4)
for epoch in range(EPOCHS):
    for x, y in train:
        # print(x.shape, y.shape)
        train_step(x, y, epoch)
        # End epoch

    ### save weights ###
    if not epoch % NUM_CHECKPOINTS_DIV:
        model.save_weights('./checkpoints/{}/{}'.format(current_time, save_c))
        save_c += 1



### TESTING ###
test_accuracy = tf.keras.metrics.Accuracy(name='test_accuracy')
# we will only test for one epoch.
for i, (x, y) in enumerate(test):

    test_accuracy.update_state(y, model(x))

    with test_summary_writer.as_default():
        tf.summary.scalar('test_accuracy', test_accuracy.result(), step=i)
    # End epoch