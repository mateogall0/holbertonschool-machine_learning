#!/usr/bin/env python3
"""
Expectation Maximization
"""


import numpy as np
initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    """
    Performs the expectation maximization for a GMM
    """
    if (not isinstance(verbose, bool) or
            not isinstance(tol, float) or
            not isinstance(iterations, int) or iterations < 1):
        return None, None, None, None, None

    pi, m, S = initialize(X, k)
    previous = []
    i = 0
    while i <= iterations:
        if pi is None or m is None or S is None:
            return None, None, None, None, None
        g, lhood = expectation(X, pi, m, S)
        if i == iterations:
            if verbose:
                print("Log Likelihood after {} iterations: {}".format(
                    i, round(lhood, 5)))
            break
        if len(previous) and np.abs(previous[-1] - lhood) <= tol:
            if verbose:
                print("Log Likelihood after {} iterations: {}".format(
                    i, round(lhood, 5)))
            break
        previous.append(lhood)
        pi, m, S = maximization(X, g)
        if verbose and (i % 10 == 0 or i + 1 % iterations == 0):
            print("Log Likelihood after {} iterations: {}".format(
                i, round(lhood, 5)))
        i += 1

    return pi, m, S, g, lhood
