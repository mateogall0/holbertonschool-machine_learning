#!/usr/bin/env python3
"""
Definiteness
"""


import numpy as np


def definiteness(matrix):
    """
    Calculates the definiteness of a matrix
    """
    if not isinstance(matrix, np.ndarray):
        raise TypeError('matrix must be a numpy.ndarray')

    try:
        eigenvalues = np.linalg.eigvals(matrix)
    except Exception:
        return
    positive_count = sum(eigenvalues > 0)
    negative_count = sum(eigenvalues < 0)
    n = len(eigenvalues)

    if positive_count == n:
        return "Positive definite"
    if positive_count > 0 and negative_count == 0:
        return "Positive semi-definite"
    if negative_count == n:
        return "Negative definite"
    if positive_count == 0 and negative_count > 0:
        return "Negative semi-definite"
    if positive_count > 0 and negative_count > 0:
        return "Indefinite"
