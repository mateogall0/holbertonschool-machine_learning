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
    # create a dense layer
    layer = tf.layers.Dense(units=n, kernel_initializer=tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG"))
    
    # apply batch normalization to the dense layer
    layer = tf.layers.BatchNormalization()(layer(prev))
    
    # add trainable parameters gamma and beta
    gamma = tf.Variable(tf.ones([n]))
    beta = tf.Variable(tf.zeros([n]))
    layer = gamma * layer + beta
    
    # apply activation function
    if activation is not None:
        layer = activation(layer)
    
    return layer