#!/usr/bin/env python3
"""
    one_hot_encode
"""


import numpy as np


def one_hot_decode(one_hot):
    """
        Converts a one-hot matrix into a vector of labels
    """
    if type(one_hot) != np.ndarray:
        return None

    one_decode = np.ndarray(shape=(one_hot.shape[1]), dtype=int)

    trnsp = one_hot.T

    for j, item in enumerate(trnsp):
        for i, it in enumerate(item):
            if it == 1.:
                one_decode[j] = i
                break
        else:
            return None

    return one_decode
