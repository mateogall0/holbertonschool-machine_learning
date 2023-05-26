#!/usr/bin/env python3
"""
DenseNet-121
"""


import tensorflow.keras as K
dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    """
    Builds the DenseNet-121 architecture as described in Densely Connected
    Convolutional Networks:

    growth_rate -- growth rate
    compression -- compression factor
    You can assume the input data will have shape (224, 224, 3)
    All convolutions should be preceded by Batch Normalization and a rectified
    linear activation (ReLU), respectively
    All weights should use he normal initialization
    Returns: the keras model
    """
    initializer = K.initializers.he_normal()
    input_data = K.Input(shape=(224, 224, 3))
    filters = 2 * growth_rate

    batch0 = K.layers.BatchNormalization()(input_data)
    activation0 = K.layers.ReLU()(batch0)
    conv0 = K.layers.Conv2D(filters=filters,
                            kernel_size=(7, 7),
                            strides=(2, 2),
                            padding="same",
                            kernel_initializer=initializer)(activation0)

    pool0 = K.layers.MaxPooling2D(pool_size=(3, 3),
                                  strides=(2, 2),
                                  padding="same")(conv0)

    dense_block1, filters = dense_block(pool0, filters,
                                        growth_rate, 6)
    transition_layer1, filters = transition_layer(dense_block1,
                                                  filters, compression)
    dense_block2, filters = dense_block(transition_layer1,
                                        filters, growth_rate, 12)
    transition_layer2, filters = transition_layer(dense_block2,
                                                  filters, compression)
    dense_block3, filters = dense_block(transition_layer2,
                                        filters, growth_rate, 24)
    transition_layer3, filters = transition_layer(dense_block3,
                                                  filters, compression)
    dense_block4, filters = dense_block(transition_layer3,
                                        filters, growth_rate, 16)
    pool1 = K.layers.AveragePooling2D(
        pool_size=(7, 7),
    )(dense_block4)
    softmax = K.layers.Dense(
        1000,
        activation='softmax',
        kernel_initializer=initializer
    )(pool1)

    return K.models.Model(inputs=input_data, outputs=softmax)
