#!/usr/bin/env python3
"""
    Convolution with padding
"""


import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """
        Performs a convolution on grayscale images with custom padding

        images -- numpy.ndarray with shape (m, h, w) containing multiple
        grayscale images
            m -- number of images
            h -- height in pixels of the images
            w -- width in pixels of the images
        kernel -- numpy.ndarray with shape (kh, kw) containing the kernel for
        the convolution
            kh -- height of the kernel
            kw -- width of the kernel
        padding -- tuple of (ph, pw)
            ph -- padding for the height of the image
            pw -- padding for the width of the image
        the image should be padded with 0's
        You are only allowed to use two for loops; any other loops of any kind
        are not allowed
        Returns: a numpy.ndarray containing the convolved images
    """
    m, h, w = images.shape
    kh, kw = kernel.shape

    oh = h - kh + 1
    ow = w - kw + 1

    output = np.zeros((m, oh, ow))

    # loop over each pixel
    for i in range(oh):
        for j in range(ow):
            image_patches = images[:, i:i+kh, j:j+kw]
            output[:, i, j] = np.sum(image_patches * kernel, axis=(1, 2))
    ph, pw = padding
    return np.pad(output, ((0, 0), (ph, ph), (pw, pw)))
