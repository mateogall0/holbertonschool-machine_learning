#!/usr/bin/env python3
from tensorflow import keras as K
import tensorflow.python.ops.numpy_ops.np_config as config
config.enable_numpy_behavior()


def shear_image(image, intensity):
    return K.preprocessing.image.random_shear(
        image, intensity, row_axis=0, col_axis=1, channel_axis=2
    )
