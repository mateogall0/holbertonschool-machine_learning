#!/usr/bin/env python3
"""
Determinant
"""


def determinant(matrix):
    """
    Calculates the determinant of a matrix
    """
    if matrix == [[]]:
        return 1
    if not isinstance(matrix, list) or len(matrix) < 1 or\
       any(not isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")
    num_rows = len(matrix)
    if num_rows == 0 or any(len(row) != num_rows for row in matrix):
        raise ValueError("matrix must be a square matrix")

    if len(matrix) == 1:
        return matrix[0][0]

    if len(matrix) == 2:
        a = matrix[0][0]
        b = matrix[0][1]
        c = matrix[1][0]
        d = matrix[1][1]
        return a * d - b * c

    if len(matrix) == 3:
        a, b, c = matrix[0][0:3]
        d, e, f = matrix[1][0:3]
        g, h, i = matrix[2][0:3]
        return (
            a * (e*i - f*h) - b * (d*i - f*g) + c * (d*h - e*g)
        )
