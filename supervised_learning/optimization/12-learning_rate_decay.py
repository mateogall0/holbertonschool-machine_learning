#!/usr/bin/env python3
"""
    Learning Rate Decay Upgraded
"""


import tensorflow as tf


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
        Creates a learning rate decay operation in tensorflow
        using inverse time decay

        alpha -- original learning rate
        decay_rate -- weight used to determine the rate at which alpha will
        decay
        global_step -- number of passes of gradient descent that have elapsed
        decay_step -- number of passes of gradient descent that
        should occur before alpha is decayed further
    """
    # Define learning rate decay function
    learning_rate = tf.train.exponential_decay(initial_learning_rate,
                                               global_step,
                                               decay_steps,
                                               decay_rate,
                                               staircase=True)

    # Define optimizer and training operation
    optimizer = tf.train.AdamOptimizer(learning_rate)
    return optimizer.minimize(loss, global_step=global_step)
