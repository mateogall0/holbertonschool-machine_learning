#!/usr/bin/env python3
"""Module"""


def cat_matrices2D(mat1, mat2, axis=0):
    """
    Concatenates two matrices along a specific axis
    """
    if len(mat1) != len(mat2):
        return None
    for i in mat1:
        for j in mat2:
            if len(i) != len(j):
                return None
    if axis == 0:
        return mat1 + mat2
    new = []
    try:
        for i in range(len(mat1)):
            new.append(mat1[i] + mat2[i])
    except Exception:
        return None
    return new
