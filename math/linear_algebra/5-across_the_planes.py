#!/usr/bin/env python3
"""Module"""


def add_matrices2D(mat1, mat2):
    """
    Adds two matrices element-wise
    """
    if len(mat1[0]) != len(mat2[0]):
        return None
    a = [[] for _ in mat1]
    for x in range(len(a)):
        for y in range(len(mat2[x])):
            a[x].append(mat1[x][y] + mat2[x][y])
    return a
