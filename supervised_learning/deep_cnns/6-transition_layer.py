#!/usr/bin/env python3
"""
Transition Layer
"""


import tensorflow.keras as K


def transition_layer(X, nb_filters, compression):
    """
    Builds a transition layer as described in Densely Connected Convolutional
    Networks:

    X -- output from the previous layer
    nb_filters -- integer representing the number of filters in X
    compression -- compression factor for the transition layer
    Your code should implement compression as used in DenseNet-C
    All weights should use he normal initialization
    All convolutions should be preceded by Batch Normalization and a rectified
    linear activation (ReLU), respectively
    Returns: The output of the transition layer and the number of filters
    within the output, respectively
    """
    X = K.layers.BatchNormalization()(X)

    X = K.layers.Activation('relu')(X)

    nb_filters = int(nb_filters * compression)

    X = K.layers.Conv2D(nb_filters, (1, 1), strides=(1, 1), padding='same',
                        kernel_initializer='he_normal')(X)

    X = K.layers.AveragePooling2D()(X)

    return X, nb_filters
