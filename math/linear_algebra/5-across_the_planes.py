#!/usr/bin/env python3
"""Module"""


def add_matrices2D(mat1, mat2):
    """
    Adds two matrices element-wise
    """
    a = []
    if (len(mat1[0]) != len(mat2[0])) or mat1 == []:
        return None
    try:
        for x in range(len(mat1)):
            for y in range(len(mat2)):
                a.append(mat1[x][y] + mat2[x][y])
    except Exception:
        return None
    return [a[:-2], a[2:]]
