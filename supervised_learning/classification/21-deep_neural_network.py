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
        if not all(map(lambda item: item >= 0, layers)):
            raise TypeError('layers must be a list of positive integers')
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        for i in range(1, self.L + 1):
            if i == 1:
                self.__weights["W" + str(i)] = \
                    np.random.randn(layers[i - 1], nx) * np.sqrt(2 / nx)
            else:
                self.__weights["W" + str(i)] = \
                    np.random.randn(layers[i - 1], layers[i - 2]) * \
                    np.sqrt(2 / layers[i - 2])
            self.__weights["b" + str(i)] = np.zeros((layers[i - 1], 1))

    @property
    def L(self):
        return self.__L

    @property
    def cache(self):
        return self.__cache

    @property
    def weights(self):
        return self.__weights

    def sigmoid(self, x):
        """
            Sigmoid activation function
        """
        return 1 / (1 + np.exp(-x))

    def forward_prop(self, X):
        """
            Calculates the forward propagation of the neural network
        """
        self.__cache['A0'] = X

        for i in range(1, self.L + 1):
            W = self.weights['W' + str(i)]
            b = self.weights['b' + str(i)]
            Aprv = self.cache['A' + str(i - 1)]
            Z = np.matmul(W, Aprv) + b  # Weighted input Z matmuling and adding
            A = self.sigmoid(Z)  # Apllies sigmoid activation to Z
            self.__cache['A' + str(i)] = A  # Saves the activated output A

        return A, self.cache

    def cost(self, Y, A):
        """
            Calculates the cost of the model using logistic regression
        """
        m = Y.shape[1]
        return -(np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))) / m

    def evaluate(self, X, Y):
        """
            Evaluates the neural networ's predictions
        """
        # Perform forward propagation to obtain the predicted output
        A, _ = self.forward_prop(X)

        # Convert predicted probabilities to binary predictions (1 or 0)
        prediction = np.where(A > 0.5, 1, 0)

        # Calculate the cost using logistic regression formula
        m = Y.shape[1]  # Number of examples
        cost = -(np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))) / m

        return prediction, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """
            Calculates one pass of gradient descent on the neural network
        """
        m = Y.shape[1]  # Number of examples

        # Compute the gradient of the cost with respect to A[L]
        dZ = self.cache['A' + str(self.L)] - Y
        # Compute the gradient of the cost with respect to W[L]
        dW = np.matmul(dZ, self.cache['A' + str(self.L - 1)].T) / m
        # Compute the gradient of the cost with respect to b[L]
        db = np.sum(dZ, axis=1, keepdims=True) / m
        # Update W[L]
        self.weights['W' + str(self.L)] -= alpha * dW
        # Update b[L]
        self.weights['b' + str(self.L)] -= alpha * db

        for i in range(self.L - 1, 0, -1):
            # Compute the gradient of the cost with respect to A[i]
            dZ = np.matmul(self.weights['W' + str(i + 1)].T, dZ) * (self.sigmoid(self.cache['A' + str(i)]))
            # Compute the gradient of the cost with respect to W[i]
            dW = np.matmul(dZ, self.cache['A' + str(i - 1)].T) / m
            # Compute the gradient of the cost with respect to b[i]
            db = np.sum(dZ, axis=1, keepdims=True) / m
            # Update W[i]
            self.weights['W' + str(i)] -= alpha * dW
            # Update b[i]
            self.weights['b' + str(i)] -= alpha * db