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
    # Create a Dense layer with kernal initializer set to FAN_AVG
    dense_layer = tf.layers.Dense(units=n, kernel_initializer=tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG"))
    
    # Compute the mean and variance of the inputs
    mean, variance = tf.nn.moments(prev, axes=[0])
    
    # Define trainable parameters gamma and beta as vectors of 1s and 0s respectively
    gamma = tf.Variable(tf.ones([n]))
    beta = tf.Variable(tf.zeros([n]))
    
    # Use batch normalization to normalize the inputs
    norm = tf.nn.batch_normalization(prev, mean, variance, beta, gamma, 1e-8)
    
    return activation(norm) 
