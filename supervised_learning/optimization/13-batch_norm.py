#!/usr/bin/env python3
"""
    Batch Normalization
"""


import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """
        Normalizes an unactivated output of a
        neural network using batch normalization

        Z -- numpy.ndarray of shape (m, n) that should be normalized
            m -- number of data points
            n -- number of features in Z
        gamma -- numpy.ndarray of shape (1, n) containing the scales
        beta -- numpy.ndarray of shape (1, n) containing the offsets
        epsilon -- small number used to avoid division by zero
    """
    # Compute the mean and variance of each feature
    mu = np.mean(Z, axis=0)
    sigma2 = np.var(Z, axis=0)

    # Normalize Z using the batch mean and variance, gamma, and beta
    Z_norm = (Z - mu) / np.sqrt(sigma2 + epsilon)
    return gamma * Z_norm + beta
