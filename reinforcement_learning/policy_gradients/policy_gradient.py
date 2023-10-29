#!/usr/bin/env python3

import numpy as np

def policy(matrix, weight):
    weighted_matrix = np.dot(matrix, weight)
    policy_matrix = np.exp(weighted_matrix) / np.sum(np.exp(weighted_matrix),
                                                     axis=1, keepdims=True)

    return policy_matrix
