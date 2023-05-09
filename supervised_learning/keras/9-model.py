#!/usr/bin/env python3
"""
    Save and Load Model
"""


import tensorflow.keras as K


def save_model(network, filename):
    """
        network -- model to save
        filename -- path of the file that the model should be saved to
        Returns: None
    """
    network.save(filename)


def load_model(filename):
    """
        filename -- path of the file that the model should be loaded from
        Returns: the loaded model
    """
    return K.models.load_model(filename)
