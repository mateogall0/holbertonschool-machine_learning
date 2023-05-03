#!/usr/bin/env python3
"""
    Forward Propagation with Dropout
"""


import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """
        Conducts forward propagation using Dropout

        X -- numpy.ndarray of shape (nx, m) containing the input data for
        the network
            nx -- number of input features
            m -- number of data points
        weights -- dictionary of the weights and biases of the neural network
        L -- number of layers in the network
        keep_prob -- probability that a node will be kept

        All layers except the last should use the tanh activation function
        The last layer should use the softmax activation function
        Returns: a dictionary containing the outputs of each layer and the
        dropout mask used on each layer
    """
    cache = {}

    cache['A0'] = X
    for l in range(1, L+1):
        W = weights['W' + str(l)]
        b = weights['b' + str(l)]

        A_prev = cache['A' + str(l - 1)]

        # compute the linear transformation of the previous layer
        Z = np.dot(W, A_prev) + b

        # apply the activation function, except for the last layer
        if l < L:
            A = np.tanh(Z)  # tanh
            # Generate a binary mask to drop out some nodes
            D = np.random.rand(A.shape[0], A.shape[1])
            D = (D < keep_prob).astype(int)

            # Apply the mask to the output of the current layer
            A = np.multiply(A, D)

            # Normalize the output of the current layer
            A /= keep_prob

            # Store the mask for backpropagation
            cache['D' + str(l)] = D
        else:
            A = np.exp(Z) / np.sum(np.exp(Z), axis=0, keepdims=True)  # softmax
        cache['A' + str(l)] = A

    return cache
