#!/usr/bin/env python3
"""
Mean and Covariance
"""


import numpy as np


def mean_cov(X):
    """
    Calculates the mean and covariance of a data set
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        raise TypeError('X must be a 2D numpy.ndarray')
    n, _ = X.shape
    if n < 2:
        raise ValueError('X must contain multiple data points')

    mean = np.mean(X, axis=0)
    mean = np.reshape(mean, (1, mean.shape[0]))
    cov = X - mean
    cov = np.dot(cov.T, cov)
    cov = cov / (n - 1)

    return mean, cov
