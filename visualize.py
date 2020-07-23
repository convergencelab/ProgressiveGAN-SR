from ann_visualizer.visualize import ann_viz
import tensorflow as tf
from ProGANVanilla import *


@tf.function
def vis():
    model = Prog_Discriminator()
    # input_shape = (16, 4, 4, 3)
    # x = tf.random.normal(input_shape)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # model(x, training=True)
    ann_viz(model, title="My first neural network")

vis()