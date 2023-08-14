#!/usr/bin/env python3
"""
Bidirectional Cell Forward
"""


import numpy as np


class BidirectionalCell:
    """
    Bidirectional Cell
    """
    def __init__(self, i, h, o):
        self.Whf = np.random.normal(size=(i + h, h))
        self.Whb = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(2 * h, o))
        self.bhf = np.zeros((1, h))
        self.bhb = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        Performs forward propagation for one time step
        """
        x_concat = np.concatenate((h_prev, x_t), axis=1)
        h_next = np.matmul(x_concat, self.Whf) + self.bhf
        h_next = np.tanh(h_next)

        return h_next

    def backward(self, h_next, x_t):
        """
        Performs backward propagation for one time step
        """
        x_concat = np.concatenate((h_next, x_t), axis=1)
        h_back = np.matmul(x_concat, self.Whb) + self.bhb
        h_back = np.tanh(h_back)

        return h_back

    def output(self, H):
        """
        Calculates all outputs for the RNN
        """
        t, m, _ = H.shape
        o = self.by.shape[-1]
        Y = np.zeros((t, m, o))

        for step in range(t):
            Y[step] = np.matmul(H[step], self.Wy) + self.by
            Y[step] = np.exp(Y[step]) / np.sum(
                np.exp(Y[step]), axis=1, keepdims=True)
        return Y
