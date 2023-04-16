#!/usr/bin/env python3
"""
    defines a deep neural network performing binary classification
"""


import numpy as np


class DeepNeuralNetwork:
    """
        Deep Neural Network class
    """
    def __init__(self, nx, layers):
        if type(nx) != int:
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        if type(layers) != list or len(layers) == 0:
            raise TypeError('layers must be a list of positive integers')
        if not all(map(lambda item: isinstance(item, int), layers)):
            raise ValueError('layers must be a list of positive integers')
        self.L = len(layers)
        self.cache = {}
        self.weights = {}

        for i in range(1, self.L + 1):
            if i == 1:
                self.weights["W" + str(i)] = \
                    np.random.randn(layers[i - 1], nx) * np.sqrt(2 / nx)
            else:
                self.weights["W" + str(i)] = \
                    np.random.randn(layers[i - 1], layers[i - 2]) * \
                    np.sqrt(2 / layers[i - 2])
            self.weights["b" + str(i)] = np.zeros((layers[i - 1], 1))
