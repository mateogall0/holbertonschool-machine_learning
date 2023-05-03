#!/usr/bin/env python3
"""
    Create a Layer with L2 Regularization
"""


import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """
        Creates a tensorflow layer that includes L2 regularization

        prev -- tensor containing the output of the previous layer
        n -- number of nodes the new layer should contain
        activation -- activation function that should be used on the layer
        lambtha -- L2 regularization parameter
        Returns: the output of the new layer
    """
    # Initialize the kernel with random values
    kernel_initializer = tf.keras.initializers.glorot_uniform()

    # Create a regularizer object using the L2 regularization parameter
    regularizer = tf.keras.regularizers.l2(lambtha)

    # Create a dense layer with the specified number of nodes and activation
    # function
    layer = tf.keras.layers.Dense(n, activation=activation,
                                  kernel_initializer=kernel_initializer,
                                  kernel_regularizer=regularizer)

    # Apply the layer to the previous tensor to obtain the output tensor
    output = layer(prev)

    return output
