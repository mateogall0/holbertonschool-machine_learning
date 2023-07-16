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
    k, = pi.shape
    n, d = X.shape

    g = np.zeros((k, n))
    for i in range(k):
        diff = X - m[i]
        inv_S = np.linalg.inv(S[i])
        exp = -0.5 * np.sum(diff @ inv_S * diff, axis=1)
        denominator = np.sqrt((2 * np.pi) ** d * np.linalg.det(S[i]))

        P = np.maximum(np.exp(exp) / denominator, 1e-300)
        g[i] = pi[i] * P

    g /= np.sum(g, axis=0)

    lhood = np.sum(np.sum(np.log(g), axis=1))

    return g, lhood
