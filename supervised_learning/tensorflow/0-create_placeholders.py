#!/usr/bin/env python3
"""
    Placeholders
"""


import tensorflow as tf


def create_placeholders(nx, classes):
    """
        Returns two placeholders
    """
    x = tf.placeholder(tf.float32, shape=(None, nx), name="x")
    y = tf.placeholder(tf.float32, shape=(None, classes), name="y")
    return x, y
