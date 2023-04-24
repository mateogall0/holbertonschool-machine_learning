#!/usr/bin/env python3
"""
    Normalize
"""


import numpy as np


def normalize(X, m, s):
    """
        Normalizes a matrix

        X is the numpy.ndarray to normalize
        d is the number of data points
        nx is the number of features
        m is a ndarray, contains the mean of all features of X
        s is a ndarray, contains the standard deviation of all features of X
    """
    return (X - m) / s
