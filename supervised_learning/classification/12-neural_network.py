#!/usr/bin/env python3
"""
    defines a neural network with one hidden
    layer performing binary classification
"""


import numpy as np


class NeuralNetwork:
    """
        Neural Network class
    """
    def __init__(self, nx, nodes):
        if type(nx) != int:
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        if type(nodes) != int:
            raise TypeError('nodes must be an integer')
        if nodes < 1:
            raise ValueError('nodes must be a positive integer')
        self.__W1 = np.random.normal(size=(nodes, nx))
        self.__b1 = np.zeros(shape=(nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.normal(size=(1, nodes))
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        return self.__W1

    @property
    def b1(self):
        return self.__b1

    @property
    def A1(self):
        return self.__A1

    @property
    def W2(self):
        return self.__W2

    @property
    def b2(self):
        return self.__b2

    @property
    def A2(self):
        return self.__A2

    def sigmoid(self, x):
        """
            Sigmoid activation function
        """
        return 1 / (1 + np.exp(-x))

    def forward_prop(self, X):
        """
            Calculates the forward propagation of the neurons
        """
        z1 = np.dot(self.W1, X) + self.b1
        self.__A1 = self.sigmoid(z1)

        z2 = np.dot(self.W2, self.A1) + self.b2
        self.__A2 = self.sigmoid(z2)
        return self.__A1, self.__A2

    def cost(self, Y, A):
        """
            Calculates the cost of the model using logistic regression
        """
        m = Y.shape[1]
        return -(np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))) / m

    def evaluate(self, X, Y):
        """
            Evaluates the neural network's predictions
        """
        A1, A2 = self.forward_prop(X)
        A = A1 * A2
        prediction = np.where((A >= 0.5), 1, 0)[:1:]
        cost2 = self.cost(Y, A2)
        return prediction, cost2
