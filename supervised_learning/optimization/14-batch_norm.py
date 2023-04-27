#!/usr/bin/env python3
"""
    Batch Normalization Upgraded
"""


import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """
        Creates a batch normalization layer for a
        neural network in tensorflow

        prev -- activated output of the previous layer
        n -- number of nodes in the layer to be created
        activation -- activation function that should
        be used on the output of the layer
    """
    # Using tf.layers.Dense as the base layer with variance_scaling_initializer
    layer = tf.layers.Dense(units=n, kernel_initializer=tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG"))

    # Creating gamma and beta trainable parameters initialized as vectors of 1 and 0 respectively
    gamma = tf.Variable(initial_value=tf.ones([n]), name='gamma')
    beta = tf.Variable(initial_value=tf.zeros([n]), name='beta')

    # Calculating the mean and variance of the layer inputs
    mean, variance = tf.nn.moments(prev, axes=[0])

    # Creating the batch normalization layer
    batch_norm = tf.nn.batch_normalization(prev, mean, variance, beta, gamma, epsilon=1e-8)

    # Applying activation function to the output of batch normalization layer
    return activation(batch_norm)
