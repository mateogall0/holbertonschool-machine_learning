#!/usr/bin/env python3

import numpy as np

def policy(matrix, weight):
    weighted_matrix = np.dot(matrix, weight)
    policy_matrix = np.exp(weighted_matrix) / np.sum(np.exp(weighted_matrix),
                                                     axis=1, keepdims=True)

    return policy_matrix

def policy_gradient(state, weight):
    policy_probabilities = policy(state, weight)

    action = np.random.choice(np.arange(policy_probabilities.shape[1]), p=policy_probabilities[0])

    gradient = state[0] - np.dot(state[0], policy_probabilities[0])

    return action, gradient
