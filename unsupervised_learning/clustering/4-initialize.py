#!/usr/bin/env python3
"""
Initialize
"""


import numpy as np
kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    """
    Initializes variables for a Gaussian Mixture Model
    """
    try:
        if len(X.shape) != 2:
            return None, None, None
        return (np.full((k,), 1/k),
                kmeans(X, k)[0],
                np.tile(np.eye(X.shape[1]), (k, 1, 1)))
    except Exception:
        return None, None, None
