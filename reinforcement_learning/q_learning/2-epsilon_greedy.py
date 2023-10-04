#!/usr/bin/env python3

import numpy as np

def epsilon_greedy(Q, state, epsilon):
    p = np.random.uniform(0, 1)
    if p > epsilon:
        action = np.argmax(Q[state,:])
    else:
        action = np.random.randint(0, Q.shape[1])
    return action
