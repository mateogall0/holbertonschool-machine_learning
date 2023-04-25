#!/usr/bin/env python3
"""
    Moving average
"""


import numpy as np


def moving_average(data, beta):
    """
        Calculates the weighted moving
        average of a data set

        Formula:
        MA = (value1 + value2 + ... + valueN) / N

        Where:

        value1 to valueN are the prices of the asset for the last n periods
        N is the number of periods used in the moving average calculation
    """
    moving_averages = []
    v = 0  # initialize exponentially weighted average to zero
    for i, d in enumerate(data):
        v = beta * v + (1 - beta) * d  # update exponentially weighted average
        v_corrected = v / (1 - beta**(i+1))  # apply bias correction
        moving_averages.append(v_corrected)
    return moving_averages
