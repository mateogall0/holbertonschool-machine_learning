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

    def _factorial(self, n):
        """
            Factorial function
            for simplification
        """
        if n == 0:
            return 1
        else:
            return n * self._factorial(n-1)

    def _binomial_coefficient(self, n, k):
        """
            Binomial coefficient
        """
        numerator = self._factorial(n)
        denominator = self._factorial(k) * self._factorial(n-k)
        return numerator // denominator

    def pmf(self, k):
        """
            Calculates the value of the PMF for a given number of “successes”
            P(X=k) = (n choose k) * p^k * (1-p)^(n-k)
            n choose k = n! / (k! * (n-k)!)
        """
        k = int(k)
        if k < 0 or k > self.n:
            return 0
        else:
            coefficient = self._binomial_coefficient(self.n, k)
            success_prob = self.p ** k
            failure_prob = (1 - self.p) ** (self.n - k)
            return coefficient * success_prob * failure_prob

    def cdf(self, k):
        """
            Calculates the value of the CDF for a given number of “successes”
            F(k;n,p) = Σ(i=0 to k) (n choose i) * p^i * (1-p)^(n-i)
        """
        k = int(k)
        if k < 0 or k > self.n:
            return 0
        cdf = 0
        for i in range(k + 1):
            b = self._binomial_coefficient(self.n, i)
            cdf += b * self.p**i * (1 - self.p)**(self.n - i)
        return cdf
