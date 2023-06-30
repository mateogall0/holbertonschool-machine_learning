#!/usr/bin/env python3
"""
Multivariate Normal distribution
"""


import numpy as np


class MultiNormal:
    """
    MultiNormal class definition
    """

    def __init__(self, data):
        if not isinstance(data, np.ndarray) or \
           len(data.shape) != 2:
            raise TypeError('data must be a 2D numpy.ndarray')
        d, n = data.shape
        if n < 2:
            raise ValueError('data must contain multiple data points')
        self.mean = np.mean(data, axis=1, keepdims=True)
        centered_data = data - self.mean
        self.cov = np.matmul(centered_data, centered_data.T) / (n - 1)
