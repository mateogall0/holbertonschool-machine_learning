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
    # Creates an initializer for the layer's weight matrix using the variance
    # scaling initializer. The "FAN_AVG" mode is chosen, which is a
    # recommended mode for initializing weights in deep neural networks.
    # This initializer helps with maintaining the variance of activations and
    # gradients throughout the network
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")

    # An L2 regularizer is created using the l2_regularizer function from
    # tf.contrib.layers. The lambtha parameter is used as the
    # regularization strength. L2 regularization penalizes the model for
    # having large weight values, helping to prevent overfitting.
    reg = tf.contrib.layers.l2_regularizer(lambtha)

    # This line creates a dense (fully connected) layer using the
    # tf.layers.Dense function. It specifies the number of nodes in the
    # layer as n, the activation function as activation, the weight
    # initializer as init, and the weight regularizer as reg. The name
    # parameter provides a name for the layer.
    layer = tf.layers.Dense(n,
                            activation=activation,
                            kernel_initializer=init,
                            name="layer",
                            kernel_regularizer=reg)
    return layer(prev)
