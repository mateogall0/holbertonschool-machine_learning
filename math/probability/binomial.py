#!/usr/bin/env python3
"""
    Binomial:
    represents a binomial distribution
"""


class Binomial:
    """
        class Binomial
    """
    e = 2.7182818285
    pi = 3.1415926536

    def __init__(self, data=None, n=1, p=0.5):
        if data is None:
            self.n = int(n)
            self.p = float(p)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            mean = sum(data) / len(data)
            variance = sum((x - mean)**2 for x in data) / (len(data))
            self.n = round(mean**2 / (mean - variance))
            self.p = mean / self.n
        if self.n <= 0:
            raise ValueError("n must be a positive value")
        if self.p <= 0 or self.p >= 1:
            raise ValueError("p must be greater than 0 and less than 1")
