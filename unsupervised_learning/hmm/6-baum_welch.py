#!/usr/bin/env python3
"""
The Baum-Welch Algorithm
"""


import numpy as np


def forward(Observation, Emission, Transition, Initial):
    """
    Performs the forward algorithm for a hidden markov model
    """
    T, = Observation.shape
    N, _ = Emission.shape

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


def baum_welch(Observations, Transition, Emission, Initial, iterations=1000):
    """
    Performs the Baum-Welch algorithm for a hidden markov model
    """
    T, = Observations.shape
    M, N = Emission.shape

    for _ in range(iterations):
        P, alpha = forward(Observations, Emission, Transition, Initial)
        _, beta = backward(Observations, Emission, Transition, Initial)
        P = np.sum(alpha[:, T - 1])

        xi = np.zeros((M, M, T - 1))
        gamma = np.zeros((M, T))

        for t in range(T - 1):
            for i in range(M):
                for j in range(M):
                    xi[i, j, t] = alpha[i, t] * Transition[
                        i, j] * Emission[j, Observations[t + 1]]
            xi[:, :, t] /= np.sum(xi[:, :, t])
            gamma[:, t] = np.sum(xi[:, :, t], axis=1)

        gamma[:, T - 1] = alpha[:, T - 1] / P
        Initial = gamma[:, 0].reshape(-1, 1)

        for i in range(M):
            for j in range(M):
                Transition[i, j] = np.sum(xi[i, j, :]) / np.sum(gamma[i, :-1])

        for i in range(M):
            for k in range(N):
                mask = (Observations == k)
                Emission[i, k] = np.sum(gamma[i, mask]) / np.sum(gamma[i, :])

    return Transition, Emission
