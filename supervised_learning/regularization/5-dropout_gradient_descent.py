#!/usr/bin/env python3
"""
    Gradient Descent with Dropout
"""


import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """
        Updates the weights of a neural network with Dropout regularization
        using gradient descent

        Y -- one-hot numpy.ndarray of shape (classes, m) that contains the
        correct labels for the data
            classes -- number of classes
            m -- number of data points
        weights -- dictionary of the weights and biases of the neural network
        cache -- dictionary of the outputs and dropout masks of each layer of
        the neural network
        alpha -- learning rate
        keep_prob -- probability that a node will be kept
        L -- number of layers of the network

        All layers use thetanh activation function except the last, which uses
        the softmax activation function
        The weights of the network should be updated in place
    """
    # Compute the gradients for the last layer
    dZ = cache["A" + str(L)] - Y
    dW = np.dot(dZ, cache["A" + str(L - 1)].T) / Y.shape[1]
    db = np.sum(dZ, axis=1, keepdims=True) / Y.shape[1]
    dA_prev = np.dot(weights["W" + str(L)].T, dZ)

    # Update the last layer's weights
    weights["W" + str(L)] -= alpha * dW
    weights["b" + str(L)] -= alpha * db

    # Loop over the remaining layers, backpropagating and updating the weights
    for l in range(L - 1, 0, -1):
        dA = dA_prev * (cache["D" + str(l)] / keep_prob)
        dZ = dA * (1 - np.power(cache["A" + str(l)], 2))
        dW = np.dot(dZ, cache["A" + str(l - 1)].T) / Y.shape[1]
        db = np.sum(dZ, axis=1, keepdims=True) / Y.shape[1]
        dA_prev = np.dot(weights["W" + str(l)].T, dZ)

        # Update the weights of this layer
        weights["W" + str(l)] -= alpha * dW
        weights["b" + str(l)] -= alpha * db
