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
    if C.shape[0] != C.shape[1]:
        raise ValueError('C must be a 2D square matrix')
    return np.corrcoef(C, rowvar=False)
