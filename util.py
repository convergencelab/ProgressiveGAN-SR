from PIL import Image
import numpy as np
# import tensorflow_datasets as tfds
import tensorflow as tf

def downsample(img, new_dims):
    """
    down sample img
    :param img: np.array img
    :param new_dims: (n, n) tuple of downsample target
    :return: downsampled img
    """
    # convert to pil Image
    img = Image.fromarray(img)
    # convert img
    img = img.resize(new_dims)

    # new axis to feed into model
    return np.array(img)[np.newaxis, ...]
"""
def preprocess(img_dict, lr_dim, upscale_factor=2):
    
    preprocess an image to be lr hr pair for a given dim
    :param img: full size img
    :param lr_dim: dims for low res
    :param upscale_factor: upscale factor for hr
    :return: lr hr pair
   

    img = img_dict['image']
    print(img)
    img_dict['image'] = tf.image.resize(img, lr_dim)

    print(img_dict['image'])
    #low_res = downsample(img, lr_dim)
    #high_res = downsample(img, (l*upscale_factor for l in lr_dim))
    return img_dict
"""