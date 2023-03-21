#!/usr/bin/env python3


def matrix_shape(matrix):
    shape = []
    if type(matrix[0]) != list:
        return [len(matrix)]
    shape += [len(matrix)]
    shape += matrix_shape(matrix[0])
    return shape
