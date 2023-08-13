#!/usr/bin/env python3
"""
RNN Cell
"""


import numpy as np


class RNNCell:
    """
    Represents a cell of a simple RNN
    """
    def __init__(self, i, h, o):
        self.Wh = np.random.normal(size=(i + h, h))
        self.bh = np.zeros((1, h))
        self.Wy = np.random.normal(size=(h, o))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        Performs forward propagation for one time step
        """
        h_x_concat = np.concatenate((h_prev, x_t), axis=1)

        h_next = np.tanh(np.dot(h_x_concat, self.Wh) + self.bh)

        output = np.dot(h_next, self.Wy) + self.by
        y = np.exp(output) / np.sum(np.exp(output), axis=1, keepdims=True)

        return h_next, y
