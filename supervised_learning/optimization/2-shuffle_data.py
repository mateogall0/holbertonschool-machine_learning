#!/usr/bin/env python3
"""
    Shuffle data
"""


import numpy as np


def shuffle_data(X, Y):
    """
        Shuffles the data points in two matrices the same way
    """
    m = X.shape[0]

    # Generate a random permutation of integers from 0 to m-1
    permutation = np.random.permutation(m)

    # Shuffle the rows of X and Y using the permutation
    shuffled_X = X[permutation]
    shuffled_Y = Y[permutation]

    # Return the shuffled X and Y matrices as a tuple
    return shuffled_X, shuffled_Y
