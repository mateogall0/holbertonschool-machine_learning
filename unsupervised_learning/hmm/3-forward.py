#!/usr/bin/env python3
"""
The Forward Algorithm
"""


import numpy as np


def forward(Observation, Emission, Transition, Initial):
    """
    Performs the forward algorithm for a hidden markov model
    """
    T, = Observation.shape
    N, M = Emission.shape

    if Transition.shape != (N, N):
        return None, None
    if Initial.shape != (N, 1):
        return None, None

    F = np.zeros((N, T))
    F[:, 0] = np.multiply(Initial[:, 0], Emission[:, Observation[0]])
    for t in range(1, T):
        for j in range(N):
            F[j, t] = np.sum(F[:, t-1] * Transition[:, j]
                             ) * Emission[j, Observation[t]]
    return np.sum(F[:, T-1]), F
