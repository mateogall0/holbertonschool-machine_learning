#!/usr/bin/env python3
"""
    Exponential
    Represents an exponential distribution
"""


class Exponential:
    """
        Exponential class
    """
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
