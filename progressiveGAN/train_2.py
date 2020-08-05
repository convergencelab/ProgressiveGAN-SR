from ProgressiveGANweightcarry import *
import tensorflow.keras.backend as K
import numpy as np
growth_phase = 1
LR_input_size = (1, 4, 4, 3)
num_filters = 32
pg = progressive_generator_phase(growth_phase,
                                 LR_input_size,
                                 num_filters)

pg.build(LR_input_size)
trainable_count1 = np.sum([K.count_params(w) for w in pg.trainable_weights])
#for l in trainable_count1:
    #print(l)
conv1 = ((8*8*32))*32
conv2 = ((3*3*32))*32
conv1 += 4*conv2
conv1 += ((32*3*3))*16
conv1 += ((3*3*16))*3

print(trainable_count1, conv1)