#!/usr/bin/env python3
"""Module"""


def cat_matrices2D(mat1, mat2, axis=0):
    """
    Concatenates two matrices along a specific axis
    """
    if matrix_shape(mat1) != matrix_shape(mat2):
        return None
    if axis == 0:
        return mat1 + mat2
    new = []
    for i in range(len(mat1)):
        new.append(mat1[i] + mat2[i])
    return new

def matrix_shape(matrix):
    """
    Recursive function that
    returns a list of numbers
    that indicate the shape
    of any given matrix
    """
    shape = []
    if type(matrix[0]) != list:
        return [len(matrix)]
    shape += [len(matrix)]
    shape += matrix_shape(matrix[0])
    return shape
