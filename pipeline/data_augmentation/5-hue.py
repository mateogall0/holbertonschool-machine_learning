#!/usr/bin/env python3
import tensorflow as tf


def change_hue(image, delta):
    return tf.image.adjust_hue(image, delta)
