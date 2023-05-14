#!/usr/bin/env python3
"""
    Pooling
"""


import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """
    Performs pooling on images

    images -- numpy.ndarray with shape (m, h, w, c) containing multiple images
        m -- number of images
        h -- height in pixels of the images
        w -- width in pixels of the images
        c -- number of channels in the image
    kernel_shape -- tuple of (kh, kw) containing the kernel shape for the
    pooling
        kh -- height of the kernel
        kw -- width of the kernel
    stride -- tuple of (sh, sw)
        sh -- stride for the height of the image
        sw -- stride for the width of the image
    mode -- indicates the type of pooling
        max -- indicates max pooling
        avg -- indicates average pooling
    You are only allowed to use two for loops; any other loops of any kind are
    not allowed
    Returns: a numpy.ndarray containing the pooled images
    """
    m, h, w, c = images.shape
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
            window = images[:, hstart:hend, wstart:wend, :]

            # Compute the pooling operation
            if mode == 'max':
                pooled_value = np.amax(window, axis=(1, 2))
            else:
                pooled_value = np.mean(window, axis=(1, 2))

            output[:, i, j, :] = pooled_value

    return output
