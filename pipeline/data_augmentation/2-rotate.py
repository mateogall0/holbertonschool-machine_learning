#!/usr/bin/env python3
import tensorflow as tf

def rotate_image(image):
    return tf.image.rot90(image)
