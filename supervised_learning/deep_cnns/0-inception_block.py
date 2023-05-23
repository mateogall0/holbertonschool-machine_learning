#!/usr/bin/env python3
"""
Inception Block
"""


import tensorflow.keras as K


def inception_block(A_prev, filters):
    """
    Builds an inception block as described in Going Deeper with Convolutions
    (2014)

    A_prev -- output from the previous layer
    filters -- tuple or list containing F1, F3R, F3,F5R, F5, FPP,
    respectively:
        F1 -- number of filters in the 1x1 convolution
        F3R -- number of filters in the 1x1 convolution before the 3x3
        convolution
        F3 -- number of filters in the 3x3 convolution
        F5R -- number of filters in the 1x1 convolution before the 5x5
        convolution
        F5 -- number of filters in the 5x5 convolution
        FPP -- number of filters in the 1x1 convolution after the max pooling
    All convolutions inside the inception block should use a rectified linear
    activation (ReLU)
    Returns: the concatenated output of the inception block
    """
    F1, F3R, F3, F5R, F5, FPP = filters

    # 1x1 convolution branch
    conv1x1 = K.layers.Conv2D(
        F1, (1, 1),
        padding='same',
        activation='relu')(A_prev)

    # 3x3 convolution branch
    conv3x3_reduce = K.layers.Conv2D(F3R, (1, 1), padding='same',
                                     activation='relu')(A_prev)
    conv3x3 = K.layers.Conv2D(F3, (3, 3), padding='same',
                              activation='relu')(conv3x3_reduce)

    # 5x5 convolution branch
    conv5x5_reduce = K.layers.Conv2D(F5R, (1, 1), padding='same',
                                     activation='relu')(A_prev)
    conv5x5 = K.layers.Conv2D(F5, (5, 5), padding='same',
                              activation='relu')(conv5x5_reduce)

    # Max pooling branch
    pool = K.layers.MaxPooling2D((3, 3), strides=(1, 1),
                                 padding='same')(A_prev)
    conv_pool = K.layers.Conv2D(FPP, (1, 1), padding='same',
                                activation='relu')(pool)

    # Concatenate the outputs of the branches
    inception_output = K.layers.concatenate(
        [conv1x1, conv3x3, conv5x5, conv_pool], axis=-1
        )

    return inception_output
