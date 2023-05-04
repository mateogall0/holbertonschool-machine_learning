#!/usr/bin/env python3
"""
    Create a Layer with Dropout
"""


import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob):
    """
        Creates a layer of a neural network using dropout

        prev -- tensor containing the output of the previous layer
        n -- number of nodes the new layer should contain
        activation -- activation function that should be used on the layer
        keep_prob -- probability that a node will be kept
        Returns: the output of the new layer
    """
    dropout = tf.layers.Dropout(keep_prob)
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.Dense(n,
                            activation=activation,
                            kernel_initializer=init,
                            kernel_regularizer=dropout)
    return layer(prev)
