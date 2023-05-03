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
    # Create weight matrix for the layer with L2 regularization
    initializer = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    regularizer = tf.contrib.layers.l2_regularizer(scale=lambtha)
    w = tf.Variable(initializer([int(prev.shape[1]), n]), dtype=tf.float32, name="kernel", regularizer=regularizer)
    
    # Compute the linear transformation of the previous layer
    z = tf.matmul(prev, w)
    
    # Apply the activation function
    if activation is not None:
        a = activation(z)
    else:
        a = z
    
    return a
