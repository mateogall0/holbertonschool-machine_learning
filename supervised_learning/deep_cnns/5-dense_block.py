#!/usr/bin/env python3
"""
Dense Block
"""


import tensorflow.keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """
    Builds a dense block as described in Densely Connected Convolutional
    Networks:

    X -- output from the previous layer
    nb_filters -- integer representing the number of filters in X
    growth_rate -- growth rate for the dense block
    layers -- number of layers in the dense block
    You should use the bottleneck layers used for DenseNet-B
    All weights should use he normal initialization
    All convolutions should be preceded by Batch Normalization and a rectified
    linear activation (ReLU), respectively
    Returns: The concatenated output of each layer within the Dense Block and
    the number of filters within the concatenated outputs, respectively
    """
    concat_layers = [X]
    nb_filters_current = nb_filters

    for _ in range(layers):
        # Bottleneck layer (DenseNet-B)
        X = K.layers.BatchNormalization()(X)
        X = K.layers.ReLU()(X)
        X = K.layers.Conv2D(filters=4 * growth_rate,
                            kernel_size=(1, 1),
                            strides=(1, 1),
                            padding='same',
                            kernel_initializer=K.initializers.he_normal())(X)

        # Convolutional layer
        X = K.layers.BatchNormalization()(X)
        X = K.layers.ReLU()(X)
        X = K.layers.Conv2D(filters=growth_rate,
                            kernel_size=(3, 3),
                            strides=(1, 1),
                            padding='same',
                            kernel_initializer=K.initializers.he_normal())(X)

        # Concatenate the output with previous layers
        concat_layers.append(X)
        X = K.layers.Concatenate()(concat_layers)
        concat_layers.append(X)
        del concat_layers[0]
        del concat_layers[0]

        # Update the number of filters
        nb_filters_current += growth_rate

    return X, nb_filters_current
