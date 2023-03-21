#!/usr/bin/env python3
"""Module"""


def cat_matrices2D(mat1, mat2, axis=0):
    """
    Concatenates two matrices along a specific axis
    """
    if axis == 0:
        return mat1 + mat2
    new = []
    for i in range(len(mat2)):
        new.append(mat1[i] + mat2[i])
    return new
