#!/usr/bin/env python3
"""
The Backward Algorithm
"""


import numpy as np


def backward(Observation, Emission, Transition, Initial):
    """
    Performs the backward algorithm for a hidden markov model
    """
    T, = Observation.shape
    N, _ = Emission.shape

    B = np.zeros((N, T))
    B[:, T - 1] = 1

    for t in range(T - 2, -1, -1):
        for s in range(N):
            B[s, t] = np.sum(B[:, t + 1] * Transition[s, :] * Emission[
                :, Observation[t + 1]])

    return np.sum(Initial[:, 0] * Emission[:, Observation[0]] * B[:, 0]), B
