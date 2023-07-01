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

    def pdf(self, x):
        """
        Calculates the PDF at a data point
        """
        if not isinstance(x, np.ndarray):
            raise TypeError('x must be a numpy.ndarray')
        if len(x.shape) != 2 or x.shape[1] != 1:
            raise ValueError('x must have the shape ({d}, 1)')
        pdf = np.exp(
            -0.5 * (x - self.mean).T @
            np.linalg.inv(self.cov) @ (x - self.mean)
            )
        pdf /= np.sqrt((2 * np.pi) ** len(x) * np.linalg.det(self.cov))
        return pdf
