#!/usr/bin/env python3
"""Module"""


def cat_matrices2D(mat1, mat2, axis=0):
    """
    Concatenates two matrices along a specific axis
    """
    if axis == 0:
        return mat1 + mat2
    new = []
    if mat1 < mat2:
        length = mat2
    else:
        length = mat1
    for i in range(length):
        new.append(mat1[i] + mat2[i])
    return new
