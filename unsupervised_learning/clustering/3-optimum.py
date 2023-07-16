#!/usr/bin/env python3
"""
Optimize K
"""


import numpy as np
kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """
    Tests for the optimum number of clusters by variance
    """
    try:
        if kmax is None:
            kmax = int(np.sqrt(len(X) / 2))
        if kmin < 1 or kmax < kmin:
            return None, None

        results = []
        d_vars = []
        min_var = 0
        for k in range(kmin, kmax, 1):
            centroids, clusters = kmeans(X, k, iterations)
            results.append((clusters, centroids))
            var = variance(X, centroids)
            if len(d_vars) == 0:
                min_var = 0
            d_vars.append(min_var - var)

        return results, d_vars
    except Exception:
        return None, None
