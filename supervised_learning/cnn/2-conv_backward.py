#!/usr/bin/env python3
"""
Convolutional Back Prop
"""


import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """
    Performs back propagation over a convolutional layer of a neural network

    dZ is a numpy.ndarray of shape (m, h_new, w_new, c_new) containing the
    partial derivatives with respect to the unactivated output of the
    convolutional layer
        m -- number of examples
        h_new -- height of the output
        w_new -- width of the output
        c_new -- number of channels in the output
    A_prev -- numpy.ndarray of shape (m, h_prev, w_prev, c_prev) containing
    the output of the previous layer
        h_prev -- height of the previous layer
        w_prev -- width of the previous layer
        c_prev -- number of channels in the previous layer
    W -- numpy.ndarray of shape (kh, kw, c_prev, c_new) containing the kernels
    for the convolution
        kh -- filter height
        kw -- filter width
    b -- numpy.ndarray of shape (1, 1, 1, c_new) containing the biases applied
    to the convolution
    padding -- string that is either same or valid, indicating the type of
    padding used
    stride -- tuple of (sh, sw) containing the strides for the convolution
        sh -- stride for the height
        sw -- stride for the width
    Returns: the partial derivatives with respect to the previous layer
    (dA_prev), the kernels (dW), and the biases (db), respectively
    """
    m, h_new, w_new, c_new = dZ.shape
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, _, _ = W.shape
    sh, sw = stride

    # Initialize the derivatives with respect to the previous layer,
    # kernels, and biases
    dA_prev = np.zeros_like(A_prev)
    dW = np.zeros_like(W)
    db = np.zeros_like(b)

    if padding == "same":
        pad_h = np.ceil((h_prev * sh - h_new + kh - sh) // 2)
        pad_w = np.ceil((w_prev * sw - w_new + kw - sw) // 2)
        A_prev_pad = np.pad(A_prev, ((0, 0), (pad_h, pad_h), (pad_w, pad_w),
                                     (0, 0)), mode="constant")
    elif padding == "valid":
        pad_h = pad_w = 0
        A_prev_pad = A_prev

    for i in range(m):
        a_prev_pad = A_prev_pad[i]
        dA_prev_pad = np.zeros_like(a_prev_pad)

        for h in range(h_new):
            for w in range(w_new):
                for c in range(c_new):
                    vert_start = h * sh
                    vert_end = vert_start + kh
                    horiz_start = w * sw
                    horiz_end = horiz_start + kw

                    a_slice = a_prev_pad[vert_start:vert_end,
                                         horiz_start:horiz_end, :]

                    dA_prev_pad[vert_start:vert_end,
                                horiz_start:horiz_end,
                                :] += W[:, :, :, c] * dZ[i, h, w, c]
                    dW[:, :, :, c] += a_slice * dZ[i, h, w, c]
                    db[:, :, :, c] += dZ[i, h, w, c]

        if padding == "same":
            dA_prev[i, :, :, :] = dA_prev_pad[pad_h:h_prev + pad_h,
                                              pad_w:w_prev + pad_w, :]
        elif padding == "valid":
            dA_prev[i, :, :, :] = dA_prev_pad

    return dA_prev, dW, db
