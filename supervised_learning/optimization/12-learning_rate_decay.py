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



        If staircase=True, the learning rate will decay in a stepwise
        fashion, meaning that the learning rate will be reduced by a
        factor of decay_rate every decay_step steps. This is useful
        when you want to apply a larger decay to the learning rate at
        specific intervals, such as at the end of an epoch or after a
        certain number of steps.

        If staircase=False, the learning rate will decay continuously,
        meaning that the learning rate will be decayed by a factor of
        decay_rate for every step, which can be useful when you want a
        more gradual and continuous decay.
    """
    return tf.train.inverse_time_decay(
        alpha,
        global_step=global_step,
        decay_steps=decay_step,
        decay_rate=decay_rate,
        staircase=True
    )
