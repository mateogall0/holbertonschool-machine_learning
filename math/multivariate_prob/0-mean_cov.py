#!/usr/bin/env python3
"""
Mean and Covariance
"""


import numpy as np


def mean_cov(X):
    """
    Calculates the mean and covariance of a data set
    """
    if X.ndim != 2:
        raise TypeError('X must be a 2D numpy.ndarray')
    if X.shape[0] < 2:
        raise ValueError('X must contain multiple data points')

    mean = np.mean(X, axis=0)
    cov = np.cov(X, rowvar=False)
    return mean, cov
