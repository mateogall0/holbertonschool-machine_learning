#!/usr/bin/env python3
"""
PDF
"""


import numpy as np


def pdf(X, m, S):
    """
    Calculates the probability density function of a Gaussian distribution
    """
    try:
        _, d = X.shape
        inv_S = np.linalg.inv(S)
        det_S = np.linalg.det(S)
        diff = X - m
        exp = -0.5 * np.sum(diff @ inv_S * diff, axis=1)
        denominator = np.sqrt((2 * np.pi) ** d * det_S)
        return np.maximum(np.exp(exp) / denominator, 1e-300)
    except Exception:
        pass
