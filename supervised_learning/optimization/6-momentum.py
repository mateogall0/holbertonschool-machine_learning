#!/usr/bin/env python3
"""
    Momentum Upgraded
"""


import tensorflow as tf


def create_momentum_op(loss, alpha, beta1):
    """
        Creates the training operation for a neural
        network in tensorflow using the gradient
        descent with momentum optimization algorithm

        loss -- the loss of the network
        alpha -- the learning rate
        beta1 -- the momentum weight
    """
    optimizer = tf.train.MomentumOptimizer(learning_rate=alpha, momentum=beta1)
    train_op = optimizer.minimize(loss)
    return train_op