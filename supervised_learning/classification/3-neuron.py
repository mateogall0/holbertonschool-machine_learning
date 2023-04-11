#!/usr/bin/env python3
"""
    Module Neuron
"""


import numpy as np


class Neuron:
    """
        Neuron class
    """

    def __init__(self, nx):
        if type(nx) != int:
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        self.__W = np.random.normal(size=(1, nx))
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        return self.__W

    @property
    def b(self):
        return self.__b

    @property
    def A(self):
        return self.__A

    def sigmoid(self, x):
        """
            Sigmoid activation function
        """
        return 1 / (1 + np.exp(-x))

    def forward_prop(self, X):
        """
            Calculates the forward propagation of the neurons
        """
        x = np.dot(self.W, X) + self.b
        self.__A = self.sigmoid(x)
        return self.A

    def cost(self, Y, A):
        """
            Calculates the cost of the model using logistic regression
        """
        m = Y.shape[1]
        return -(np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))) / m

