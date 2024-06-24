#!/usr/bin/env python3
import tensorflow as tf


def flip_image(image):
    return tf.image.flip_left_right(image)
