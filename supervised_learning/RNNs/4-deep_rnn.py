#!/usr/bin/env python3
"""
Deep RNN
"""


import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """
    Performs forward propagation for a deep RNN
    """
    t, m, i = X.shape
    ln = len(rnn_cells)
    h = h_0.shape[-1]

    H = np.zeros((t + 1, ln, m, h))
    H[0] = h_0

    Y = np.zeros((t, m, rnn_cells[-1].by.shape[-1]))
    for t_step in range(t):
        for layer in range(ln):
            if t_step == 0:
                prev_h = h_0[layer]
            else:
                prev_h = H[t_step, layer]
            if layer == 0:
                new_h, y = rnn_cells[layer].forward(prev_h, X[t_step])
            else:
                new_h, y = rnn_cells[
                    layer].forward(prev_h, H[t_step, layer - 1])
            H[t_step + 1, layer] = new_h
            if layer == ln - 1:
                Y[t_step] = y

    return H[1:], Y
