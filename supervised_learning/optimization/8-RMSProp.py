#!/usr/bin/env python3
"""
    RMSProp Upgraded
"""


import tensorflow as tf


def create_RMSProp_op(loss, alpha, beta2, epsilon):
    """
        Creates the training operation for a neural
        network in tensorflow using the
        RMSProp optimization algorithm

        loss -- the loss of the network
        alpha -- learning rate
        beta2 -- RMSProp weight
        epsilon -- small number to avoid division by zero
    """
    optimizer = tf.train.RMSPropOptimizer(learning_rate=alpha,
                                          decay=beta2,
                                          epsilon=epsilon)
    return optimizer.minimize(loss)
