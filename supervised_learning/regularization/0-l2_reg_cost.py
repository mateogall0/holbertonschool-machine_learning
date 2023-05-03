#!/usr/bin/env python3
"""
    L2 Regularization Cost
"""


import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """
        Calculates the cost of a neural network with L2 regularization

        cost -- cost of the network without L2 regularization
        lambtha -- regularization parameter
        weights -- dictionary of the weights and biases (numpy.ndarrays)
        of the neural network
        L -- number of layers in the neural network
        m -- number of data points used
        Returns: the cost of the network accounting for L2 regularization

        L2 regularization term = lambtha * ||w||^2
    """
    # Compute the L2 regularization term
    l2_term = 0
    for l in range(1, L+1):
        W = weights['W' + str(l)]
        l2_term += np.sum(np.square(W))  # ||w||^2
    l2_term *= (lambtha / (2 * m))  # lambtha * ||w||^2

    # adding a penalty for large weight values to the original cost function
    return cost + l2_term
