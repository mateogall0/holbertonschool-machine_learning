#!/usr/env python3
"""
    Momentum
"""


import numpy as np


def update_variables_momentum(alpha, beta1, var, grad, v):
    """
        Updates a variable using the gradient
        descent with momentum optimization algorithm

        Calculate the momentum term:
        v = beta1 * v + (1 - beta1) * grad
        Here, grad is the gradient of the loss function
        with respect to the weights at time step t.

        Update the weights using the momentum term:
        var = var - alpha * v
        Here, var represents the weights at time step t.
    """
    return (
        var - alpha * v,
        beta1 * v + (1 - beta1) * grad
    )
