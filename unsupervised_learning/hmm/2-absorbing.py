#!/usr/bin/env python3
"""
Absorbing
"""


import numpy as np


def absorbing(P):
    """
    Determines if a markov chain is absorbing
    """
    return P[0][0] == 1 and all(i[0] != 1 for i in P[1:])
