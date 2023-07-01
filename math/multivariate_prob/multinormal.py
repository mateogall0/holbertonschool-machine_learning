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
        _, n = data.shape
        if n < 2:
            raise ValueError('data must contain multiple data points')
        self.mean = np.mean(data, axis=1, keepdims=True)
        centered_data = data - self.mean
        self.cov = np.matmul(centered_data, centered_data.T) / (n - 1)

    def pdf(self, x):
        """
        Calculates the PDF at a data point
        f(x) = (1 / (sqrt((2 * π)^k * det(Σ)))) *
              exp(-0.5 * (x - μ)^T * Σ^(-1) * (x - μ))
        """
        if not isinstance(x, np.ndarray):
            raise TypeError('x must be a numpy.ndarray')
        if len(x.shape) != 2 or x.shape[1] != 1:
            raise ValueError('x must have the shape ({}, 1)'.format(
                x.shape[0])
                )
        try:
            k = self.mean.shape[0]
            det_cov = np.linalg.det(self.cov)
            inv_cov = np.linalg.inv(self.cov)

            exponent = -0.5 * np.dot(np.dot((x - self.mean).T, inv_cov),
                                     (x - self.mean))
            coefficient = 1 / (np.sqrt((2 * np.pi) ** k * det_cov))

            pdf = coefficient * np.exp(exponent)
        except Exception:
            raise ValueError('x must have the shape ({}, 1)'.format(
                self.mean.shape[0])
                )
        return pdf[0][0]
