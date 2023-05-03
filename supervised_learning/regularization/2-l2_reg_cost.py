#!/usr/bin/env python3
"""
    L2 Regularization Cost
"""


import tensorflow as tf


def l2_reg_cost(cost):
    """
        Calculates the cost of a neural network with L2 regularization

        cost -- tensor containing the cost of the network without L2 regularization
        Returns: a tensor containing the cost of the network accounting for L2 regularization
    """
    lambtha = 0.01 # regularization parameter
    weights = tf.trainable_variables() # dictionary containing the weights of the network

    l2_reg = 0
    for weight in weights:
        l2_reg += tf.reduce_sum(tf.square(weight))

    l2_reg *= (lambtha / (2 * len(weights)))

    cost_with_reg = cost + l2_reg

    return cost_with_reg
