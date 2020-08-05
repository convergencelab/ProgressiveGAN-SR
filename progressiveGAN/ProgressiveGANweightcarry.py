"""
Progressive Growing Gan

Grows from 4x4 to 8x8 and so on...
rather than a dynamically growing architecture,
train on one phase, carry over the weights to next phase...
may be able to reduce the size of the actual architecture we are working on

for generator:
    initialize next phase with:
        simple mapping of weights: for example 2x the last weights
        consider the stats behind what I was pondering last week: activity levels and
        the importance of them.


"""
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, UpSampling2D, AveragePooling2D, LeakyReLU
from tensorflow.keras.activations import softmax
import tensorflow as tf
import numpy as np
from Layers import *

filters = [32, 64, 128, 256, 512, 1024]

class progressive_generator_phase(Model):
    """
    new generator for each size increase,

    previous work is carried over via the weight intialization
    of this network.

    phases to be implemented:
    4x4 -> 8x8
    8x8 -> 16x16
    16x16 -> 32x32
    32x32 -> 64x64
    32x32 -> 64x64
    64x64 -> 128x128
    128x128 -> 256x256
    """
    def __init__(self,
                 growth_phase,
                 LR_input_size,
                 num_filters,
                 leakyrelu_alpha=0.2,
                 kernel_initializer='he_normal',
                 **kwargs
                 ):

        # call the parent constructor
        super(progressive_generator_phase, self).__init__(**kwargs)
        self.leakyrelu_alpha = leakyrelu_alpha
        self.LR_input_size = LR_input_size
        self.kernel_initializer = kernel_initializer


        # to enable reduction of filter size
        self.growth_phase = growth_phase
        self.num_filters = num_filters
        # conv 4x4, input block
        self.input_conv = Conv2DEQ(
                              input_shape=self.LR_input_size,
                              filters=self.num_filters,
                              kernel_size=(4, 4),
                              padding='same',
                              kernel_initializer=self.kernel_initializer
                              )

        self.acti = LeakyReLU(alpha=self.leakyrelu_alpha)

        # first conv
        # reduction of filters process:
        self.conv0 = Conv2DEQ(filters=self.reduce_filters(5),
                              kernel_size=(3, 3),
                              padding='same',
                              kernel_initializer=self.kernel_initializer
                              )

        self.act0 = LeakyReLU(alpha=self.leakyrelu_alpha)
        self.pixel_w_norm0 = PixelNormalization()

        self.conv1 = Conv2DEQ(filters=self.reduce_filters(4),
                              kernel_size=(3, 3),
                              padding='same',
                              kernel_initializer=self.kernel_initializer
                              )

        self.act1 = LeakyReLU(alpha=self.leakyrelu_alpha)
        self.pixel_w_norm1 = PixelNormalization()


        self.conv2 = Conv2DEQ(filters=self.reduce_filters(3),
                              kernel_size=(3, 3),
                              padding="same",
                              kernel_initializer=self.kernel_initializer
                              )
        self.act2 = LeakyReLU(alpha=0.2)
        self.pixel_w_norm2 = PixelNormalization()

        self.conv3 = Conv2DEQ(filters=self.reduce_filters(2),
                              kernel_size=(3, 3),
                              padding="same",
                              kernel_initializer=self.kernel_initializer
                              )

        self.act3 = LeakyReLU(alpha=0.2)
        self.pixel_w_norm3 = PixelNormalization()

        self.upspl = UpSampling2D()
        self.conv4 = Conv2DEQ(filters=self.reduce_filters(1),
                                   kernel_size=(3, 3),
                                   padding='same',
                                   kernel_initializer='he_normal'
                                   )

        self.act4 = LeakyReLU(alpha=0.2)
        self.conv5 = Conv2DEQ(filters=self.reduce_filters(0),
                                   kernel_size=(3, 3),
                                   padding='same',
                                   kernel_initializer='he_normal'
                                   )

        self.act5 = LeakyReLU(alpha=0.2)
        self.RGB_out = Conv2DEQ(filters=3,
                                kernel_size=(1, 1),
                                padding='same',
                                kernel_initializer='he_normal'
                                )

    def reduce_filters(self, i):
        if self.growth_phase > i:
            return self.num_filters/2
        else:
            return self.num_filters

    def return_weights(self):
        weights = []
        for l in self.layers:
            if isinstance(l, Conv2DEQ):
                weights.append(l.get_weights())
        return weights

    def initialize_weights(self, weights):
        weights = iter(weights)
        for l in self.layers:
            if isinstance(l, Conv2DEQ):
                l.set_weights(next(weights))


    def call(self, x):
        x = self.input_conv(x)
        x = self.acti(x)
        x = self.conv0(x)
        x = self.act0(x)
        x = self.pixel_w_norm0(x)
        x = self.conv1(x)
        x = self.act1(x)
        x = self.pixel_w_norm1(x)
        x = self.conv2(x)
        x = self.act2(x)
        x = self.pixel_w_norm2(x)
        x = self.conv3(x)
        x = self.act3(x)
        x = self.pixel_w_norm3(x)
        x = self.upspl(x)
        x = self.conv4(x)
        x = self.act4(x)
        x = self.conv5(x)
        x = self.act5(x)
        x = self.RGB_out(x)
        return x

class Prog_Discriminator(Model):
    """
    No normnalization according to paper.

    Critic -> WGAN-GP loss
    No clipping -> gradient penalty instead
    """
    def __init__(self,
                 #clip_constraint,
                 leakyrelu_alpha=0.2,
                 **kwargs
                 ):
        # call the parent constructor
        super(Prog_Discriminator, self).__init__(**kwargs)
        #self.clip_constraint = ClipConstraint(clip_constraint)
        self.leakyrelu_alpha = leakyrelu_alpha

        # intialize with 512
        self.num_filters = 512

        # to enable reduction of filter size
        self.growth_phase = 0

        ### Construct base model ###
        # Input for first growth
        self.input_conv = Conv2DEQ(filters=self.num_filters,
                                    kernel_size=(1, 1),
                                    padding='same',
                                    kernel_initializer='he_normal',
                                    name="input"
                                    #kernel_constraint=self.clip_constraint
                                 )
        self.input_act = LeakyReLU(alpha=0.2)
        self.input_dnsmpl = AveragePooling2D()

        # top to be filled with dis blocks
        self.dis_blocks = []

        # minibatch std dev layer
        self.MinibatchStdev = MinibatchStdev()

        # conv 3x3

        self.conv1 = Conv2DEQ(filters=512,
                            kernel_size=(3, 3),
                            padding='same',
                            kernel_initializer='he_normal',
                            name="conv1"
                            #kernel_constraint=self.clip_constraint
                            )
        self.act1 = LeakyReLU(alpha=leakyrelu_alpha, name="act1")
        # self.pixel_w_norm1 = PixelNormalization()

        # conv 4x4 (1x1 dims)
        self.conv2 = Conv2DEQ(  filters=512,
                                kernel_size=(4, 4),
                                padding='same',
                                strides=(512, 512),
                                kernel_initializer='he_normal',
                                name="conv2"
                                # kernel_constraint=self.clip_constraint
                            )
        self.act2 = LeakyReLU(alpha=leakyrelu_alpha, name="act2")
        # self.pixel_w_norm2 = PixelNormalization()
        # self.droput = tf.keras.layers.Dropout(0.3)
        # dense output layer
        self.flatten = Flatten(name="flatten")
        self.dense = Dense(1, name="dense")

        #self.act3 = softmax
        # weighted output
        self.weighted_sum = WeightedSum(name="weighted_sum")

        # intialize with growth period
        self.grow()

    def set_ws_alpha(self, alpha):
        self.weighted_sum.set_alpha(alpha)

    def grow(self):
        num_filters = self.num_filters
        # reduction in filters occurs after 3rd phase
        decrease_filters = self.growth_phase > 2
        # insert new dis block to front of list
        self.dis_blocks.insert(0,
            dis_block(num_filters,
                      decrease_filters,
                      #self.clip_constraint
                      )
        )

        self.growth_phase += 1
        if decrease_filters:
            self.num_filters = int(self.num_filters / 2)

        # disable following blocks input section
        if self.growth_phase > 1:
            self.dis_blocks[1].is_top = False
            # disable training for input layers in block.
            self.dis_blocks[1].deactivate_input()

    def call(self, inputs):
        # input block (this may be an erraneous implementation #
        # inputs will potentially need to grow with the model #
        x = inputs

        if self.growth_phase > 1:
            # straight pass
            x_prime = self.dis_blocks[0](x)
            x_prime = self.dis_blocks[1](x_prime)
            # pass through old block as if it is input
            self.dis_blocks[1].is_top = True
            x = self.dis_blocks[1](x)
            self.dis_blocks[1].is_top = False

            # pass through all layers except last layer
            for i, block in enumerate(self.dis_blocks[2:]):
                x = block(x)
                x_prime = block(x_prime)

        else:
            # straight pass, nothing to pass through x
            x_prime = self.dis_blocks[0](x)
            # input activation for first growth phase ( old model )
            x = self.input_conv(x)
            x = self.input_act(x)

        x = self.input_dnsmpl(x)

        # straight pass cont'd
        x_prime = self.MinibatchStdev(x_prime)
        x_prime = self.conv1(x_prime)
        x_prime = self.act1(x_prime)
        x_prime = self.conv2(x_prime)

        x_prime = self.act2(x_prime)
        x_prime = self.flatten(x_prime)
        x_prime = self.dense(x_prime)

        # old pass
        x = self.MinibatchStdev(x)
        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.act2(x)
        x = self.flatten(x)
        x = self.dense(x)

        # fade in two outputs
        x = self.weighted_sum([x, x_prime])

        return x