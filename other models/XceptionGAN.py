"""
using xception as generator
"""
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np
from imagedict import *
### Hyperparameters ###
gen_lr=0.01
dis_lr=0.01
beta_1=0.5
beta_2=0.9
epsilon=10e-8
batch_size = 16
up_sample = 2

### data ###

# train_ds, test_ds, train_ds_size = prepare_and_upscale(lr_dim=(128, 128))
data, info = tfds.load('resisc45', split="train", with_info=True)
tfds.show_examples(data, info)
train_ds = data.batch(batch_size)
train_ds = tfds.as_numpy(train_ds)



### adam optimizer for SGD ###
gen_optimizer = tf.keras.optimizers.Adam(lr=gen_lr,
                                         beta_1=beta_1,
                                         beta_2=beta_2,
                                         epsilon=epsilon)


model = tf.keras.applications.Xception(weights='imagenet', include_top=True)
model.compile(gen_optimizer
              )
model.build(input_shape=(batch_size, 256, 256, 3))
inp = next(iter(train_ds))['image']
# show convolutional output with no top.
out = model(inp)
out1 = out[0].numpy()
plt.imshow(inp[0])
plt.show()
# out1 = out1.reshape(2048, 8, 8) * 255.0
# fig, ax = plt.subplots(16)
winner = np.argmax(out1)
print(IMAGED[winner])
