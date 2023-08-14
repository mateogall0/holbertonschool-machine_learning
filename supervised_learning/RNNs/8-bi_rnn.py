#!/usr/bin/env python3
"""
BIRNN
"""


import numpy as np


def bi_rnn(bi_cell, X, h_0, h_t):
    """
    Performs forward propagation for a bidirectional RNN
    """
    t, m, _ = X.shape
    _, h = h_0.shape
    H_for = np.zeros((t, m, h))
    H_back = np.zeros((t, m, h))
    h_ft = h_0
    h_bt = h_t
    for step in range(t):
        x_ft = X[step]
        x_bt = X[-(step + 1)]

        h_ft = bi_cell.forward(h_ft, x_ft)
        h_bt = bi_cell.backward(h_bt, x_bt)

        H_for[step] = h_ft
        H_back[-(step + 1)] = h_bt

    H = np.concatenate((H_for, H_back), axis=-1)
    Y = bi_cell.output(H)

    return H, Y
