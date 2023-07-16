#!/usr/bin/env python3
"""
EM
"""


import numpy as np
initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    """
    Performs the expectation maximization for a GMM
    """
    pi, m, S = initialize(X, k)
    previous = None
    for i in range(iterations + 1):
        if pi is None or m is None or S is None:
            return None, None, None, None, None
        g, lhood = expectation(X, pi, m, S)
        if previous and np.abs(previous - lhood) <= tol:
            if verbose:
                print("Log Likelihood after {} iterations: {}".format(
                    i, round(lhood, 5)))
            break
        previous = lhood
        pi, m, S = maximization(X, g)
        if verbose and (i % 10 == 0 or i + 1 % iterations == 0):
            print("Log Likelihood after {} iterations: {}".format(
                i, round(lhood, 5)))

    return pi, m, S, g, lhood
