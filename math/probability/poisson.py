#!/usr/bin/env python3
"""
    Probability, Poisson
"""


class Poisson:
    """
        Poisson class:
        represents a poisson distribution
    """
    e = 2.7182818285
    pi = 3.1415926536

    def __init__(self, data=None, lambtha=1.):
        if data is None:
            if lambtha <= 0:
                raise ValueError('lambtha must be a positive value')
            self.lambtha = lambtha
        else:
            if not isinstance(data, list):
                raise TypeError('data must be a list')
            if len(data) < 2:
                raise ValueError('data must contain multiple values')
            self.lambtha = sum(data) / len(data)
        self.lambtha = float(self.lambtha)

    def pmf(self, k):
        """
            Calculates the value of the PMF for a given number of “successes”
        """
        k = int(k)
        if k < 0:
            return 0

        factorial = 1
        for i in range(1, k+1):
            factorial *= i
        return self.lambtha ** k * self.e**(-self.lambtha) / factorial

    def cdf(self, k):
        """
        Calculates the value of the CDF for a given number of “successes”
        """
        k = int(k)
        if k < 0:
            return 0
        cdf_value = 0
        for i in range(k + 1):
            cdf_value += self.pmf(i)
        return cdf_value
