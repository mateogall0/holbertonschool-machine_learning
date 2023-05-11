#!/usr/bin/env python3
"""
    Convolution with Channels
"""


import numpy as np


def convolve_channels(images, kernel, padding='same', stride=(1, 1)):
    """
    Performs a convolution on images with channels

    images -- numpy.ndarray with shape (m, h, w, c) containing multiple
    images
        m -- number of images
        h -- height in pixels of the images
        w -- width in pixels of the images
        c -- number of channels in the image
    kernel -- numpy.ndarray with shape (kh, kw, c) containing the kernel for
    the convolution
        kh -- height of the kernel
        kw -- width of the kernel
    padding -- either a tuple of (ph, pw), ‘same’, or ‘valid’
        if ‘same’, performs a same convolution
        if ‘valid’, performs a valid convolution
        if a tuple:
            ph -- padding for the height of the image
            pw -- padding for the width of the image
        the image should be padded with 0’s
    stride -- tuple of (sh, sw)
        sh -- stride for the height of the image
        sw  stride for the width of the image
    ou are only allowed to use two for loops; any other loops of any kind are
    not allowed
    Returns: a numpy.ndarray containing the convolved images
    """
    m, h, w, _ = images.shape
    kh, kw, _ = kernel.shape

    sh, sw = stride

    ph, pw = 0, 0
    if type(padding) == tuple:
        ph, pw = padding
    elif padding == 'same':
        ph = (((h - 1) * sh + kh - h) // 2) + 1
        pw = (((w - 1) * sw + kw - w) // 2) + 1

    if ph and pw:
        images = np.pad(images, ((0, 0), (ph, ph), (pw, pw), (0, 0)), mode='constant')

    oh = ((h + (2 * ph) - kh) // sh) + 1
    ow = ((w + (2 * pw) - kw) // sw) + 1
    output = np.zeros((m, oh, ow))

    # loop over each pixel
    for i in range(oh):
        for j in range(ow):
            output[:, i, j] = (
                kernel * images[:, i * sh:i * sh + kh, j * sw:j * sw + kw]
            ).sum(axis=(1, 2, 3))
    return output
