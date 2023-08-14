#!/usr/bin/env python3
"""
LSTM Cell
"""


import numpy as np


class LSTMCell:
    """LSTM unit"""
    def __init__(self, i, h, o):
        self.Wf = np.random.normal(size=(i + h, h))
        self.Wu = np.random.normal(size=(i + h, h))
        self.Wc = np.random.normal(size=(i + h, h))
        self.Wo = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(h, o))
        self.bf = np.zeros((1, h))
        self.bu = np.zeros((1, h))
        self.bc = np.zeros((1, h))
        self.bo = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def sigmoid(self, x):
        """
        Sigmoid func
        """
        return 1 / (1 + np.exp(-x))

    def forward(self, h_prev, c_prev, x_t):
        """
        performs forward propagation for one time step
        """
        concat_input = np.concatenate((h_prev, x_t), axis=1)
        ft = self.sigmoid(np.matmul(concat_input, self.Wf) + self.bf)
        ut = self.sigmoid(np.matmul(concat_input, self.Wu) + self.bu)
        c_tilde = np.tanh(np.matmul(concat_input, self.Wc) + self.bc)
        c_next = ft * c_prev + ut * c_tilde
        ot = self.sigmoid(np.matmul(concat_input, self.Wo) + self.bo)
        h_next = ot * np.tanh(c_next)
        y = np.matmul(h_next, self.Wy) + self.by

        return h_next, c_next, y
