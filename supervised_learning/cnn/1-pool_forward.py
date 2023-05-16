#!/usr/bin/env python3
"""
    Pooling Forward Prop
"""


import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Performs forward propagation over a pooling layer of a neural network

    A_prev -- numpy.ndarray of shape (m, h_prev, w_prev, c_prev) containing
    the output of the previous layer
        m -- number of examples
        h_prev -- height of the previous layer
        w_prev -- width of the previous layer
        c_prev -- number of channels in the previous layer
    kernel_shape -- tuple of (kh, kw) containing the size of the kernel for
    the pooling
        kh -- kernel height
        kw -- kernel width
    stride -- tuple of (sh, sw) containing the strides for the pooling
        sh -- stride for the height
        sw -- stride for the width
    mode -- string containing either max or avg, indicating whether to perform
    maximum or average pooling, respectively
    Returns: the output of the pooling layer
    """
    m, h, w, c = A_prev.shape
    kh, kw = kernel_shape

    sh, sw = stride

    oh = (h - kh) // sh + 1
    ow = (w - kw) // sw + 1
    output = np.zeros((m, oh, ow, c))

    for i in range(oh):
        for j in range(ow):
            # Define the window
            hstart = i * sh
            hend = hstart + kh
            wstart = j * sw
            wend = wstart + kw
            window = A_prev[:, hstart:hend, wstart:wend, :]

            # Compute the pooling operation
            if mode == 'max':
                pooled_value = np.amax(window, axis=(1, 2))
            else:
                pooled_value = np.mean(window, axis=(1, 2))

            output[:, i, j, :] = pooled_value

    return output
