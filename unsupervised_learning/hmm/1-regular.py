#!/usr/bin/env python3
"""
Regular Chain
"""


import numpy as np


def regular(P):
    """
    Determines the steady state probabilities of a regular markov chain

    πᵀP = πᵀ
    """
    n, n1 = P.shape
    if n != n1:
        return None
    if np.any(np.linalg.matrix_power(P, n) == 0):
        return None

    A = np.eye(n) - P.T
    A[-1] = np.ones(n)

    b = np.zeros(n)
    b[-1] = 1
    pi = np.linalg.lstsq(A, b, rcond=None)[0]

    if np.isnan(pi).any() or np.isinf(pi).any():
        return None

    pi /= np.sum(pi)

    return pi.reshape(1, n)
