#!/usr/bin/env python3
"""
This module contains a function that
measures the shape of an array by
returning an array with these values
"""

def matrix_shape(matrix):
    shape = []
    if type(matrix[0]) != list:
        return [len(matrix)]
    shape += [len(matrix)]
    shape += matrix_shape(matrix[0])
    return shape
