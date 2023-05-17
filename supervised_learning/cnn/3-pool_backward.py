#!/usr/bin/env python3
"""
Pooling Back Prop
"""


import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Performs back propagation over a pooling layer of a neural network

    dA -- numpy.ndarray of shape (m, h_new, w_new, c_new) containing the
    partial derivatives with respect to the output of the pooling layer
        m -- number of examples
        h_new -- height of the output
        w_new -- width of the output
        c -- number of channels
    A_prev -- numpy.ndarray of shape (m, h_prev, w_prev, c) containing the
    output of the previous layer
        h_prev -- height of the previous layer
        w_prev -- width of the previous layer
    kernel_shape -- tuple of (kh, kw) containing the size of the kernel for
    the pooling
        kh -- kernel height
        kw -- kernel width
    stride -- tuple of (sh, sw) containing the strides for the pooling
        sh -- stride for the height
        sw -- stride for the width
    mode -- string containing either max or avg, indicating whether to perform
    maximum or average pooling, respectively
    you may import numpy as np
    Returns: the partial derivatives with respect to the previous layer
    (dA_prev)
    """
    m, h_new, w_new, c_new = dA.shape
    m, h_prev, w_prev, c = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    dA_prev = np.zeros_like(A_prev)

    for i in range(m):
        for h in range(h_new):
            for w in range(w_new):
                for c in range(c_new):
                    vert_start = h * sh
                    vert_end = vert_start + kh
                    horiz_start = w * sw
                    horiz_end = horiz_start + kw

                    if mode == 'max':
                        mask = A_prev[i,
                                      vert_start:vert_end,
                                      horiz_start:horiz_end,
                                      c] == np.max(
                            A_prev[i,
                                   vert_start:vert_end,
                                   horiz_start:horiz_end,
                                   c]
                            )
                        dA_prev[i,
                                vert_start:vert_end,
                                horiz_start:horiz_end,
                                c] += mask * dA[i, h, w, c]
                    elif mode == 'avg':
                        average = dA[i, h, w, c] / (kh * kw)
                        dA_prev[i,
                                vert_start:vert_end,
                                horiz_start:horiz_end,
                                c] += np.ones((kh, kw)) * average

    return dA_prev
