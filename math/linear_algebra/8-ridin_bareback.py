#!/usr/bin/env python3
"""Module"""


def mat_mul(mat1, mat2):
    """
    Performs matrix multiplication
    """
    if len(mat1[0]) != len(mat2):
        return None
    p = len(mat2[0])
    new = [[0]*p for _ in mat1]
    for i in range(len(mat1)):
        for j in range(len(mat2)):
            for k in range(p):
                new[i][k] += mat1[i][j] * mat2[j][k]
    return new
