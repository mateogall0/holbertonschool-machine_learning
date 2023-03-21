#!/usr/bin/env python3
"""Module"""


def matrix_transpose(matrix):
    """
    Returns the transpose of a 2D matrix
    """
    return [list(i) for i in zip(*matrix)]
