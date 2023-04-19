#!/usr/bin/env python3
"""
    Forward Propagation
"""


import tensorflow as tf
create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """
        Creates the forward propagation graph
    """
    prev = x
    for i in range(len(layer_sizes)):
        with tf.variable_scope("layer" + str(i+1), reuse=tf.AUTO_REUSE):
            layer = create_layer(prev, layer_sizes[i], activations[i])
            prev = layer
    return layer
