#!/usr/bin/env python3
"""
    Sigma
"""


def summation_i_squared(n):
    """Summation"""
    if (type(n) != int):
        return
    if (n < 1):
        return 0
    return n * n + summation_i_squared(n - 1)
