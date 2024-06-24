#!/usr/bin/env python3
import tensorflow as tf


def change_brightness(image, max_delta):
    return tf.image.random_brightness(image, max_delta)