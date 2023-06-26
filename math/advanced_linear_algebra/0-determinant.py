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

    det = 0
    for i in range(len(matrix)):
        cofactor = (-1) ** i
        sub_matrix = []
        for j in range(1, len(matrix)):
            sub_matrix_row = matrix[j][:i] + matrix[j][i + 1:]
            sub_matrix.append(sub_matrix_row)
        sub_det = determinant(sub_matrix)
        det += cofactor * matrix[0][i] * sub_det

    return det
