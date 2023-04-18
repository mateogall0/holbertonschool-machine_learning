#!/usr/bin/env python3
"""
    one_hot_encode
"""


import numpy as np


def one_hot_encode(Y, classes):
    """
        Converts a numeric label vector into a one-hot matrix
    """
    if type(Y) != np.ndarray:
        return None

    if type(classes) != int or classes <= 0:
        return None
    max_label = np.max(Y)
    if max_label >= classes:
        return None

    # Create an empty one-hot encoding matrix
    one_hot = np.zeros((classes, len(Y)))
    # Loop through each example in Y and
    # set the corresponding one-hot vector
    for i in range(len(Y)):
        one_hot[Y[i], i] = 1

    return one_hot
