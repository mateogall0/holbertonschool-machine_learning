#!/usr/bin/env python3
"""
Correlation
"""


import numpy as np


def correlation(C):
    """
    Calculates a correlation matrix
    """
    if not isinstance(C, np.ndarray):
        raise TypeError('C must be a numpy.ndarray')
    if len(C.shape) != 2 or C.shape[0] != C.shape[1]:
        raise ValueError('C must be a 2D square matrix')
    diagonal = np.diag(1 / np.sqrt(np.diag(C)))
    return np.matmul(np.matmul(diagonal, C), diagonal)
