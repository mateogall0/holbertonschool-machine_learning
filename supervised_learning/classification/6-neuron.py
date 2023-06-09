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
        return -(np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))) / m

    def evaluate(self, X, Y):
        """
            Evaluates the neuron's predictions
        """
        A = self.forward_prop(X)
        prediction = np.where(A >= 0.5, 1, 0)
        cost = self.cost(Y, A)
        return prediction, cost

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """
            Calculates one pass of gradient descent on the neuron
        """
        m = Y.shape[1]
        dZ = A - Y
        dW = (1/m) * np.dot(X, dZ.T)
        db = (1/m) * np.sum(dZ)
        self.__W -= alpha * dW.T
        self.__b -= alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """
            Trains the neuron
        """
        if type(iterations) != int:
            raise TypeError('iterations must be an integer')
        if iterations < 0:
            raise ValueError('iterations must be a positive integer')
        if type(alpha) != float:
            raise TypeError('alpha must be a float')
        if alpha < 0.0:
            raise ValueError('alpha must be positive')

        for _ in range(iterations):
            A = self.forward_prop(X)
            self.cost(Y, A)
            self.gradient_descent(X, Y, A, alpha)

        return self.evaluate(X, Y)
