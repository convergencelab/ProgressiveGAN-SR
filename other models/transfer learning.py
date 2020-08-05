"""
first transfer learning to brush up on
"""

import os
import numpy
import matplotlib.pyplot as plt

import tensorflow as tf

import tensorflow_datasets as tfds

(train, validation, test), info = tfds.load('cats_vs_dogs',
                                            split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
                                            with_info=True,
                                            as_supervised=True)

IMG_SIZE = 128

label_names = info.features['label'].int2str
fig, ax = plt.subplots(1,2, figsize=(5, 10))
for i, (image, label) in enumerate(train.take(2)):
    ax[i].imshow(image)
    ax[i].set_title(label_names(label))
plt.show()

def map_img(x, y):
    img = tf.cast(x, tf.float32)
    img = (img/127.5 -1)
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
    return img, y
# map imgs
train = train.map(map_img)
validation = validation.map(map_img)
test = test.map(map_img)

BATCH_SIZE = 32
SHUFFLE_BUFFER_SIZE = 1000
# shuffle ds
train = train.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
validation = validation.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
test = test.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)

# inspect
for image_batch, label_batch in train.take(1):
   pass

print(image_batch.shape)

IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)

model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                          weights='imagenet',
                                          include_top=False
                                          )

# freeeze model
model.trainable = False

full_model = tf.keras.Sequential([
                model,
                tf.keras.layers.GlobalAveragePooling2D(),
                tf.keras.layers.Dense(1)
])

lr = 0.0001
full_model.compile(optimizer=tf.optimizers.Adam(lr=lr),
                   loss=tf.losses.BinaryCrossentropy(from_logits=True),
                   metrics=[tf.metrics.BinaryAccuracy(threshold=0.0, name='BA')])

### train ###
history = full_model.fit(train,
                    epochs=10,
                    validation_data=validation)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

fig, ax = plt.subplots(1, 2)
ax[0].plot(acc, label='acc')
ax[0].plot(val_acc, label='val_acc')
ax[0].set_title('accuracy')

ax[1].plot(loss, label='loss')
ax[1].plot(val_loss, label='loss')
ax[1].set_title('loss')

plt.legend()
plt.show()