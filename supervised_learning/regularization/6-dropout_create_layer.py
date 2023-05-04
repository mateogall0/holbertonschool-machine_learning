#!/usr/bin/env python3
"""
    Create a Layer with Dropout
"""


import numpy as np


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
    W = np.random.randn(n, prev.shape[0]) * np.sqrt(2 / prev.shape[0])
    b = np.zeros((n, 1))
    
    # Compute the linear output of the new layer
    Z = np.dot(W, prev) + b
    
    # Apply dropout to the linear output
    D = np.random.rand(Z.shape[0], Z.shape[1]) < keep_prob
    A = np.multiply(Z, D)
    A /= keep_prob

    # Apply the activation function to the output
    output = activation(A)

    return output
