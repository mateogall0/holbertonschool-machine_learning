#!/usr/bin/env python3
"""
    Adam
"""


import numpy as np


def update_variables_Adam(alpha, beta1, beta2,
                          epsilon, var, grad,
                          v, s, t):
    """
        Updates a variable in place using the
        Adam optimization algorithm

        alpha -- learning rate
        beta1 -- weight used for the first moment
        beta2 -- weight used for the second moment
        epsilon -- small number to avoid division by zero
        var -- numpy.ndarray containing the variable to be updated
        grad -- numpy.ndarray containing the gradient of var
        v -- previous first moment of var
        s -- previous second moment of var
        t -- time step used for bias correction
    """

    # Compute biased first moment estimate
    v_new = beta1 * v + (1 - beta1) * grad

    # Compute biased second moment estimate
    s_new = beta2 * s + (1 - beta2) * np.square(grad)

    # Correct bias in first and second moment estimates
    v_corrected = v_new / (1 - np.power(beta1, t))
    s_corrected = s_new / (1 - np.power(beta2, t))

    # Update variable

    return (
        var - alpha * v_corrected / (np.sqrt(s_corrected) + epsilon),
        v_new, s_new
        )
