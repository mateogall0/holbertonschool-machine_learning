#!/usr/bin/env python3
"""
PCA
"""


import numpy as np


def pca(X, var=0.95):
    """
    Performs PCA on a dataset
    """
    _, s, vh = np.linalg.svd(X)
    cumulative_variance = np.cumsum(s)
    threshold = cumulative_variance[len(cumulative_variance) - 1] * var
    mask = np.where(threshold > cumulative_variance)
    retained_variance = cumulative_variance[mask]
    num_components = len(retained_variance) + 1
    W = vh.T
    return W[:, 0:num_components]
