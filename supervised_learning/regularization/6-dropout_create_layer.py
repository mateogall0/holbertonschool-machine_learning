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
    # Initialize the weights and biases of the new layer
    W = tf.Variable(tf.random.normal([n, prev.shape[1]],
                                     stddev=tf.sqrt(2 / prev.shape[1])))
    b = tf.Variable(tf.zeros([n, 1]))

    # Compute the linear output of the new layer
    Z = tf.matmul(W, prev, transpose_b=True) + b

    # Apply dropout to the linear output
    D = tf.nn.dropout(tf.ones_like(Z), keep_prob=keep_prob)
    A = tf.multiply(Z, D)

    # Apply the activation function to the output
    output = activation(A)

    return output
