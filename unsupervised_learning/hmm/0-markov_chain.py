#!/usr/bin/env python3
"""
Markov Chain
"""


import numpy as np


def markov_chain(P, s, t=1):
    """
    Determines the probability of a markov chain being in a particular state
    after a specified number of iterations
    """
    try:
        if t < 1:
            return None
        result = s @ np.linalg.matrix_power(P, t)
        return result / np.sum(result)
    except Exception:
        return None
