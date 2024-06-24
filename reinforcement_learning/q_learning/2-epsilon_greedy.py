#!/usr/bin/env python3
import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """
    Uses epsilon-greedy to determine the next action
    """
    p = np.random.uniform(0, 1)
    if p > epsilon:
        return np.argmax(Q[state,:])
    return np.random.randint(0, Q.shape[1])
