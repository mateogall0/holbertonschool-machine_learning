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

    padding_h = max((kh - 1) // 2, kh // 2)
    padding_w = max((kw - 1) // 2, kw // 2)

    padded_images = np.pad(images, ((0, 0), (padding_h, padding_h),
                           (padding_w, padding_w)), mode='constant')

    output = np.zeros((m, h, w))

    # loop over each pixel
    for i in range(h):
        for j in range(w):
            output[:, i, j] = (kernel * padded_images[
                :, i: i + kh, j: j + kw]).sum(axis=(1, 2))

    return output
