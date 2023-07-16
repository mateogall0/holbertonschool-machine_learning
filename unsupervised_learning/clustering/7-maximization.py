#!/usr/bin/env python3
"""
Mximization
"""


import numpy as np


def maximization(X, g):
    """
    Calculates the maximization step in the EM algorithm for a GMM
    """
    try:
        n, d = X.shape
        k, _ = g.shape

        Nk = np.sum(g, axis=1)
        pi = Nk / n
        m = g @ X / Nk[:, np.newaxis]

        S = np.zeros((k, d, d))
        for i in range(k):
            diff = X - m[i]
            Wdiff = g[i, :, np.newaxis] * diff
            S[i] = Wdiff.T @ diff / Nk[i]
        return pi, m, S
    except Exception:
        return None, None
