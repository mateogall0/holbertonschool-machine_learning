#!/usr/bin/env python3
"""
    Sigma
"""

import numpy as np


def summation_i_squared(n):
    """Summation"""
    if (type(n) != int):
        return
    a = np.array(list(range(1, n)))
    s = np.power(a, 2)
    return sum(s)
