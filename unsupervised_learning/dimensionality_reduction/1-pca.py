#!/usr/bin/env python3
"""
PCA v2
"""


import numpy as np


def pca(X, ndim):
    """
    Performs PCA on a dataset
    """
    X_centered = X - np.mean(X, axis=0)
    _, _, vh = np.linalg.svd(X_centered)
    W = vh.T
    Wr = W[:, 0:ndim]
    return X_centered @ Wr
