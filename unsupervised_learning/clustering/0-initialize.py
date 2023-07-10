#!/usr/bin/env python3
"""
Initialize
"""


import numpy as np


def initialize(X, k):
    """
    Initializes cluster centroids for K-means

    X -- numpy.ndarray of shape (n, d) containing the dataset that will be
    used for K-means clustering
        n -- number of data points
        d -- number of dimensions for each data point
    k -- positive integer containing the number of clusters
    The cluster centroids should be initialized with a multivariate uniform
    distribution along each dimension in d:
    The minimum values for the distribution should be the minimum values of X
    along each dimension in d
    The maximum values for the distribution should be the maximum values of X
    along each dimension in d

    Returns: a numpy.ndarray of shape (k, d) containing the initialized
    centroids for each cluster, or None on failure
    """
    if not isinstance(k, int) or k <= 0:
        return
    try:
        min = np.min(X, axis=0)
        max = np.max(X, axis=0)

        # Obtain the centroids
        return np.random.uniform(min, max, size=(k, X.shape[1]))
    except Exception:
        pass
