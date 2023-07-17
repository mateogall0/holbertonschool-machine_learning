#!/usr/bin/env python3
"""
Bayesian Information Criterion
"""


import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """
    Finds the best number of clusters for a GMM using the Bayesian Information
    Criterion
    """
    try:
        n, d = X.shape
        if kmax is None:
            kmax = n
            if kmax >= kmin:
                return None, None, None, None
        kHistory = list(range(kmin, kmax+1))
        resultsHistory = []
        lhoodHistory = []
        bicHistory = []
        for k in range(kmin, kmax+1):
            pi, m, S, g, lhood = expectation_maximization(
                X, k, iterations, tol, verbose
            )
            p = d * k + (d * k * (d + 1) / 2) + k - 1
            BIC = p * np.log(n) - 2 * lhood

            resultsHistory.append((pi, m, S))
            lhoodHistory.append(lhood)
            bicHistory.append(BIC)

            i = np.argmin(bicHistory)
            best_k = kHistory[i]
            best_result = resultsHistory[i]

        return best_k, best_result, np.array(lhoodHistory), np.array(bicHistory)
    except Exception:
        return None, None, None, None