#!/usr/bin/env python3
"""
Expectation
"""


import numpy as np
pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """
    Calculates the expectation step in the EM algorithm for a GMM
    """
    try:
        k, = pi.shape
        n, _ = X.shape

        g = np.zeros((k, n))
        for i in range(k):
            g[i] = pi[i] * pdf(X, m[i], S[i])
        marginal = np.sum(g, axis=0)
        g /= np.sum(g, axis=0)
        lhood = np.sum(np.log(marginal))

        return g, lhood
    except Exception:
        return None, None
