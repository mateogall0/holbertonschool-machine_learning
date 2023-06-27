#!/usr/bin/env python3
"""
Inverse
"""


def inverse(matrix):
    """
    Calculates the inverse of a matrix
    """
    if not isinstance(matrix, list) or len(matrix) < 1 or\
       any(not isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")
    num_rows = len(matrix)
    if num_rows == 0 or any(len(row) != num_rows for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    det = determinant(matrix)
    if det == 0:
        return

    adj = adjugate(matrix)
    return [[
        adj[i][j] / det for j in range(num_rows)
        ] for i in range(num_rows)]


def adjugate(matrix):
    """
    Calculates the adjugate matrix of a matrix
    """
    if not isinstance(matrix, list) or len(matrix) < 1 or\
       any(not isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")
    num_rows = len(matrix)
    if num_rows == 0 or any(len(row) != num_rows for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")
    return [list(i) for i in zip(*cofactor(matrix))]


def cofactor(matrix):
    """
    Calculates the cofactor matrix of a matrix
    """
    if not isinstance(matrix, list) or len(matrix) < 1 or\
       any(not isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")
    num_rows = len(matrix)
    if num_rows == 0 or any(len(row) != num_rows for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")
    cofactor = list(matrix)
    m = minor(matrix)
    for i in range(len(m)):
        for j in range(len(m[i])):
            cofactor[i][j] = (-1) ** (i+j) * m[i][j]

    return cofactor


def minor(matrix):
    """
    Calculates the minor matrix of a matrix
    """
    if not isinstance(matrix, list) or len(matrix) < 1 or\
       any(not isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")
    num_rows = len(matrix)
    if num_rows == 0 or any(len(row) != num_rows for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")
    if len(matrix) == 1:
        return [[1]]
    minors = []
    n = len(matrix)

    for i in range(n):
        minors_row = []
        for j in range(n):
            sub_matrix = [
                row[:j] + row[j + 1:] for row_idx,
                row in enumerate(matrix) if row_idx != i
                ]
            minor = determinant(sub_matrix)
            minors_row.append(minor)
        minors.append(minors_row)

    return minors


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
