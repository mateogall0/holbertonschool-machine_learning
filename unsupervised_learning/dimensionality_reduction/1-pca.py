#!/usr/bin/env python3
"""
PCA v2
"""


import numpy as np


def pca(X, ndim):
    """
    Performs PCA on a dataset
    """
    cov_matrix = np.cov(X, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]
    selected_eigenvectors = sorted_eigenvectors[:, :ndim]
    return np.dot(X, selected_eigenvectors)
