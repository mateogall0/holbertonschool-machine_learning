#!/usr/bin/env python3
"""
Initialize
"""


import numpy as np
import tensorflow as tf


class NST:
    """
    NST
    """
    style_layers = ['block1_conv1', 'block2_conv1',
                    'block3_conv1', 'block4_conv1', 'block5_conv1']
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        if type(style_image) != np.ndarray or style_image.shape[-1] != 3:
            raise TypeError('style_image must be a numpy.ndarray with \
shape (h, w, 3)')
        if type(content_image) != np.ndarray or content_image.shape[-1] != 3:
            raise TypeError('content_image must be a numpy.ndarray with shape \
(h, w, 3)')
        if (type(alpha) != int and type(alpha) != float) or alpha <= 0:
            raise TypeError('alpha must be a non-negative number')
        if (type(beta) != int and type(beta) != float) or beta <= 0:
            raise TypeError('beta must be a non-negative number')

        tf.enable_eager_execution()
        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta

    @staticmethod
    def scale_image(image):
        if type(image) != np.ndarray or image.shape[-1] != 3:
            raise TypeError("image must be a numpy.ndarray with shape \
(h, w, 3)")
        h, w = image.shape[:2]
        max_side = max(h, w)
        scale_factor = 512 / max_side
        new_h, new_w = scale_factor * h, scale_factor * w
        resized_image = tf.constant(image, tf.int32)
        resized_image = tf.expand_dims(resized_image, axis=0)
        resized_image = tf.image.resize_images(resized_image, [new_h, new_w])
        resized_image = resized_image / 255.0
        resized_image = tf.clip_by_value(
            resized_image, clip_value_min=0, clip_value_max=1
            )
        return tf.cast(resized_image, tf.float32)
