#!/usr/bin/env python3
"""
Initialize Bayesian Optimization
"""


import numpy as np
from scipy.stats import norm
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """
    Performs Bayesian optimization on a noiseless 1D Gaussian process
    """
    def __init__(self, f, X_init, Y_init, bounds, ac_samples, l=1,
                 sigma_f=1, xsi=0.01, minimize=True):
        self.f = f
        self.gp = GP(X_init, Y_init, l, sigma_f)
        self.X_s = np.linspace(
            bounds[0], bounds[1], ac_samples).reshape(-1, 1)
        self.xsi = xsi
        self.minimize = minimize

    def acquisition(self):
        """
        Calculates the next best sample location
        """
        mu, sigma = self.gp.predict(self.X_s)
        mu = mu.reshape(-1, 1)
        sigma = sigma.reshape(-1, 1)

        if self.minimize:
            Y_sample = np.min(self.gp.Y)
        else:
            Y_sample = np.max(self.gp.Y)

        with np.errstate(divide='warn'):
            imp = Y_sample - mu - self.xsi
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0
        X_next = self.X_s[np.argmax(ei)]
        EI = ei.flatten()
        return X_next, EI

    def optimize(self, iterations=100):
        """
        Optimizes the black-box function
        """
        position = []
        for _ in range(iterations):
            X_next, ei = self.acquisition()
            Y_next = self.f(X_next)
            p = np.argmax(ei)
            if p in position:
                position.append(p)
                break
            self.gp.update(X_next, Y_next)
            position.append(p)
        if self.minimize is True:
            X_next = np.argmin(self.gp.Y)
        else:
            X_next = np.argmax(self.gp.Y)
        self.gp.X = np.delete(self.gp.X, -1, axis=0)
        return self.gp.X[X_next], self.gp.Y[X_next]
