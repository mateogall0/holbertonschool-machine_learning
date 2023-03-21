#!/usr/bin/env python3
"""Module"""


def add_arrays(arr1, arr2):
    """
    Add arrays
    """
    if len(arr1) != len(arr2):
        return None
    a = []
    for x, y in zip(arr1, arr2):
        a.append(x + y)
    return a
