#!/usr/bin/env python3
"""
The Viretbi Algorithm
"""


import numpy as np


def viterbi(Observation, Emission, Transition, Initial):
    """
    Calculates the most likely sequence of hidden states for a hidden markov
    model
    """
    T, = Observation.shape
    N, _ = Emission.shape

    Viterbi = np.zeros((N, T))
    backpointer = np.zeros((N, T))

    Viterbi[:, 0] = Initial[:, 0] * Emission[:, Observation[0]]
    for t in range(1, T):
        for s in range(N):
            temp = Viterbi[
                :, t - 1] * Transition[:, s] * Emission[s, Observation[t]]
            Viterbi[s, t] = np.max(temp)
            backpointer[s, t] = np.argmax(temp)

    path = np.zeros(T, dtype=int)
    path[T - 1] = np.argmax(Viterbi[:, T - 1])
    for t in range(T - 2, -1, -1):
        path[t] = backpointer[path[t + 1], t + 1]
    P = np.max(Viterbi[:, T - 1])

    return list(path), P
