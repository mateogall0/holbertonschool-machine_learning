#!/usr/bin/env python3
"""
    Sigma
"""


def summation_i_squared(n):
    """Summation"""
    if (type(n) != int):
        return
    return int(n * (n + 1) * (2 * n + 1) / 6)
