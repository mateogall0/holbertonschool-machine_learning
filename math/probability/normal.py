#!/usr/bin/env python3
"""
    Normal:
    represents a normal distribution
"""


class Normal:
    """
        class Normal
    """
    e = 2.7182818285
    pi = 3.1415926536

    def __init__(self, data=None, mean=0., stddev=1.):
        if stddev <= 0:
            raise ValueError('stddev must be a positive value')

        if data is None:
            self.mean = float(mean)
            self.stddev = float(stddev)
        else:
            if not isinstance(data, list):
                raise TypeError('data must be a list')
            if len(data) < 2:
                raise ValueError('data must contain multiple values')

            self.mean = sum(data) / len(data)
            s = [(x - self.mean) ** 2 for x in data]
            self.stddev = (sum(s) / (len(data) - 1)) ** 0.5

        self.mean = float(self.mean)
        self.stddev = float(self.stddev)
