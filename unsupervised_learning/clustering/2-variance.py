#!/usr/bin/env python3
"""
Variance
"""


import numpy as np


def variance(X, C):
    """
    Calculates the total intra-cluster variance for a data set
    """
    try:
        distances = np.linalg.norm(X[:, np.newaxis] - C, axis=2)
        closest_centroids = np.argmin(distances, axis=1)
        squared_distances = np.sum((X - C[closest_centroids]) ** 2, axis=1)
        var = np.sum(squared_distances)
        return var
    except Exception:
        pass
