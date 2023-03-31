#!/usr/bin/env python3
"""
    Sigma
"""


def summation_i_squared(n):
    """Summation"""
    if (type(n) != int):
        return
    return round((sum(range(n ** 2)) / n) - n)
