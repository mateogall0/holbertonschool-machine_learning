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
        b = weights['W' + str(l)]

        A_prev = cache['A' + str(l - 1)]

        # compute the linear transformation of the previous layer
        Z = np.dot(W, A_prev) + b

        # apply the activation function, except for the last layer
        if l < L:
            A = np.tanh(Z)  # tanh
            # matrix of random values between 0 and 1
            D = np.random.normal(A.shape[0], A.shape[1])
            # Binary mask D comparing values with keep_prob
            D = D < keep_prob
            # some of the activation values will be set to 0
            # while others will remain unchanged
            A *= D
            # rescaling operation to have the same expected value
            # as before the masking operation
            A /= keep_prob
            cache['D' + str(l)] = D
        else:
            A = np.exp(Z) / np.sum(np.exp(Z), axis=0, keepdims=True)  # softmax

    return cache
