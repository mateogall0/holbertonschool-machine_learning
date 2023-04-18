#!/usr/bin/env python3
"""
    defines a deep neural network performing binary classification
"""


import numpy as np
import matplotlib.pyplot as plt


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

        return prediction, self.cost(Y, A)

    def gradient_descent(self, Y, cache, alpha=0.05):
        """
            Calculates one pass of gradient descent on the neural network
        """
        weights = self.__weights
        m = Y.shape[1]
        err = cache["A" + str(self.L)] - Y

        for i in range(self.L, 0, -1):
            W, b = "W{}".format(i), "b{}".format(i)
            A = "A{}".format(i - 1)
            # Compute the gradient of bias
            bias_gradient = np.sum(err, axis=1, keepdims=True) / m
            # Calculate the gradient of the weights
            Wg = np.matmul(err, cache[A].T) / m
            err = np.matmul(weights[W].T, err)*(cache[A]*(1-cache[A]))
            weights[W] -= alpha * Wg
            weights[b] -= alpha * bias_gradient

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True, graph=True, step=100):
        """
            Trains the deep neural network
        """
        if type(iterations) != int:
            raise TypeError('iterations must be an integer')
        if iterations < 0:
            raise ValueError('iterations must be a positive integer')
        if type(alpha) != float:
            raise TypeError('alpha must be a float')
        if alpha < 0.0:
            raise ValueError('alpha must be positive')

        data = []

        for i in range(iterations + 1
                       ):
            A, cache = self.forward_prop(X)
            cost = self.cost(Y, A)
            self.gradient_descent(Y, cache, alpha)
            if verbose and (i % step == 0 or i % iterations == 0):
                data.append({i: cost})
                print(f'Cost after {i} iterations: {cost}')

        if graph and len(data) > 0:
            x = [list(d.keys())[0] for d in data]
            y = [list(d.values())[0] for d in data]
            plt.plot(x, y)
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title('Training Cost')
            plt.show()

        return self.evaluate(X, Y)
