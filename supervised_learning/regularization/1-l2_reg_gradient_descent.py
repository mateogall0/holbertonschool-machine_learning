#!/usr/bin/env python3
"""
    Gradient Descent with L2 Regularization
"""


import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
        Updates the weights and biases of a neural network using gradient
        descent with L2 regularization

        Y -- one-hot numpy.ndarray of shape (classes, m) that contains the
        correct labels for the data
            classes -- number of classes
            m -- number of data points
        weights -- dictionary of the weights and biases of the neural network
        cache -- dictionary of the outputs of each layer of the neural network
        alpha -- learning rate
        lambtha -- L2 regularization parameter
        L -- number of layers of the network


        The neural network uses tanh activations on each layer except the
        last, which uses a softmax activation
        The weights and biases of the network should be updated in place

        Formula:
        W = W - alpha * (dW + (lambtha/m) * W)
    """
    m = Y.shape[1]
    dZ = cache['A'+str(L)] - Y
    for l in range(L, 0, -1):
        A_prev = cache['A'+str(l-1)]
        W = weights['W'+str(l)]
        b = weights['b'+str(l)]
        # 1/m: scaling factor for the derivatives to account for the number of
        # data points
        dW = 1/m * np.dot(dZ, A_prev.T) + (lambtha/m)*W
        # keepdims=True: keeps the shape of the sum, so that db has the same
        # shape as b
        db = 1/m * np.sum(dZ, axis=1, keepdims=True)
        # derivative of tanh activation
        dZ = np.dot(W.T, dZ) * (1 - np.power(A_prev, 2))
        weights['W'+str(l)] -= alpha * dW
        weights['b'+str(l)] -= alpha * db
