#!/usr/bin/env python3
"""
Likelihood
"""


import numpy as np


def likelihood(x, n, P):
    """
    Calculates the likelihood of obtaining this data given various
    hypothetical probabilities of developing severe side effects

    likelihood = (n choose x) * p^x * (1 - p)^(n - x)
    """
    if not isinstance(n, int) or n < 0:
        raise ValueError('n must be a positive integer')
    if not isinstance(n, int) or x < 0:
        raise ValueError(
            'x must be an integer that is greater than or equal to 0'
            )
    if x > n:
        raise ValueError('x cannot be grater than n')
    if not isinstance(P, np.ndarray) or len(P.shape) != 1:
        raise TypeError('P must be a 1D numpy.ndarray')
    if not all(0 <= x <= 1 for x in P):
        raise ValueError("All values in P must be in the range [0, 1]")

    likelihoods = np.zeros_like(P)

    for i, p in enumerate(P):
        binomial_coefficient = np.math.factorial(n) / (
            np.math.factorial(x) * np.math.factorial(n - x)
            )
        likelihood = binomial_coefficient * (p ** x) * ((1 - p) ** (n - x))
        likelihoods[i] = likelihood

    return likelihoods
