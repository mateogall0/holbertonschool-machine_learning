#!/usr/bin/env python3
"""
    RMSProp
"""


import numpy as np


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """
        Updates a variable using the RMSProp optimization algorithm

        alpha -- learning rate
        beta2 -- decay rate for the moving average of the squared gradient
        epsilon -- small constant to avoid dividing by zero
        var -- variable to be updated
        grad -- gradient of the loss function with respect to the variable
        s -- exponentially decaying average of past squared gradients

    """
    # Compute the updated second moment using the current gradient and
    # previous second moment
    s = beta2 * s + (1 - beta2) * np.square(grad)

    # Compute the update vector
    update = grad / (np.sqrt(s) + epsilon)

    # Apply the update to the variable
    var -= alpha * update

    return var, s
