#!/usr/bin/env python3
"""
    Optimize
"""

import tensorflow.keras as K


def optimize_model(network, alpha, beta1, beta2):
    """
        Sets up Adam optimization for a keras model with categorical
        crossentropy loss and accuracy metrics

        network -- model to optimize
        alpha -- learning rate
        beta1 -- first Adam optimization parameter
        beta2 -- second Adam optimization parameter

        Returns: None
    """
    opt = K.optimizers.Adam(alpha, beta_1=beta1, beta_2=beta2)
    network.compile(loss='categorical_crossentropy', optimizer=opt,
                    metrics=['accuracy'])
