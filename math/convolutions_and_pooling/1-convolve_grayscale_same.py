#!/usr/bin/env python3
"""
    Same Convolution
"""


import numpy as np


def convolve_grayscale_same(images, kernel):
    """
        Performs a same convolution on grayscale images

        images -- numpy.ndarray with shape (m, h, w) containing multiple
        grayscale images
            m -- number of images
            h -- height in pixels of the images
            w -- width in pixels of the images
        kernel -- numpy.ndarray with shape (kh, kw) containing the kernel for
        the convolution
            kh -- height of the kernel
            kw -- width of the kernel

        if necessary, the image should be padded with 0's
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
    for i in range(1, oh):
        for j in range(1, ow):
            image_patches = images[:, i:i+kh, j:j+kw]
            output[:, i, j] = np.sum(image_patches * kernel, axis=(1, 2))
    if output.shape != images.shape:
        pad_h = (images.shape[1] - output.shape[1]) // 2
        pad_w = (images.shape[2] - output.shape[2]) // 2
        return np.pad(output, ((0, 0), (pad_h, pad_h), (pad_w, pad_w)),
                      mode='constant')
    return output
