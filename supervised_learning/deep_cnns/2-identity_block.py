#!/usr/bin/env python3
"""
Identify Blocks
"""


import tensorflow.keras as K


def identity_block(A_prev, filters):
    """
    Builds an identity block as described in Deep Residual Learning for Image
    Recognition (2015):

    A_prev -- output from the previous layer
    filters -- tuple or list containing F11, F3, F12, respectively:
        F11 -- number of filters in the first 1x1 convolution
        F3 -- number of filters in the 3x3 convolution
        F12 -- number of filters in the second 1x1 convolution
    All convolutions inside the block should be followed by batch
    normalization along the channels axis and a rectified linear activation
    (ReLU), respectively.
    All weights should use he normal initialization
    Returns: the activated output of the identity block
    """
    F11, F3, F12 = filters
    initializer = K.initializers.he_normal()

    conv1x1 = K.layers.Conv2D(
        F11, (1, 1),
        kernel_initializer=initializer,
        padding='same',
        activation='linear')(A_prev)
    batch_norm0 = K.layers.BatchNormalization()(conv1x1)
    activation0 = K.layers.ReLU()(batch_norm0)
    conv3x3 = K.layers.Conv2D(F3, (3, 3), kernel_initializer=initializer,
                              padding='same',
                              activation='linear')(activation0)
    batch_norm1 = K.layers.BatchNormalization()(conv3x3)
    activation1 = K.layers.ReLU()(batch_norm1)
    conv_pool = K.layers.Conv2D(F12, (1, 1),
                                kernel_initializer=initializer,
                                padding='same',
                                activation='linear')(activation1)
    batch_norm2 = K.layers.BatchNormalization()(conv_pool)

    add = K.layers.Add()([batch_norm2, A_prev])
    return K.layers.ReLU()(add)
