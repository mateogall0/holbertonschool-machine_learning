#!/usr/bin/env python3
"""
Identify Blocks
"""


import tensorflow.keras as K


def projection_block(A_prev, filters, s=2):
    """
    Builds an identity block as described in Deep Residual Learning for Image
    Recognition (2015):

    A_prev -- output from the previous layer
    filters -- tuple or list containing F11, F3, F12, respectively:
        F11 -- number of filters in the first 1x1 convolution
        F3 -- number of filters in the 3x3 convolution
        F12 -- number of filters in the second 1x1 convolution as well as the
        1x1 convolution in the shortcut connection
    s -- stride of the first convolution in both the main path and the
    shortcut connection
    All convolutions inside the block should be followed by batch
    normalization along the channels axis and a rectified linear activation
    (ReLU), respectively.
    All weights should use he normal initialization
    Returns: the activated output of the projection block
    """
    F11, F3, F12 = filters
    initializer = K.initializers.he_normal()

    conv0 = K.layers.Conv2D(
        F11, (1, 1),
        strides=s,
        kernel_initializer=initializer,
        padding='same',
        activation='linear')(A_prev)
    batch0 = K.layers.BatchNormalization()(conv0)
    activation0 = K.layers.ReLU()(batch0)
    conv1 = K.layers.Conv2D(F3, (3, 3),
                            kernel_initializer=initializer,
                            padding='same',
                            activation='linear')(activation0)
    batch1 = K.layers.BatchNormalization()(conv1)
    activation1 = K.layers.ReLU()(batch1)
    conv2 = K.layers.Conv2D(
        F12,
        (1, 1),
        kernel_initializer=initializer,
        padding='same',
        activation='linear'
    )(activation1)
    conv3 = K.layers.Conv2D(
        F12,
        (1, 1),
        strides=s,
        kernel_initializer=initializer,
        padding='same',
        activation='linear'
    )(A_prev)
    batch2 = K.layers.BatchNormalization()(conv2)
    batch3 = K.layers.BatchNormalization()(conv3)
    add = K.layers.Add()([batch2, batch3])
    return K.layers.ReLU()(add)
