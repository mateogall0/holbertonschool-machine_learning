#!/usr/bin/env python3
"""
    Exponential
    Represents an exponential distribution
"""


class Exponential:
    """
        Exponential class
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
            self.lambtha = len(data) / sum(data)
        self.lambtha = float(self.lambtha)

    def pdf(self, x):
        """
            Calculates the value of the PDF for a given time period
        """
        if x < 0:
            return 0
        return self.lambtha * self.e ** (-self.lambtha * x)

    def cdf(self, x):
        """
            Calculates the value of the CDF for a given time period
        """
        x = int(x)
        if x < 0:
            return 0
        return 1 - self.e ** (-self.lambtha * x)
