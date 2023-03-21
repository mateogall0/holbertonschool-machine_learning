#!/usr/bin/env python3
"""Module"""


def add_matrices2D(mat1, mat2):
    """
    Adds two matrices element-wise
    """
    a = []
    if mat1 == [] or (len(mat1[0]) != len(mat2[0])):
        return None
    if len(mat1) == 0 or (len(mat1) != len(mat2)):
        return None
    for x in range(len(mat1)):
        for y in range(len(mat2)):
            a.append(mat1[x][y] + mat2[x][y])
    return [a[:-2], a[2:]]