#!/usr/bin/env python3
"""
    Save and Load Configuration
"""


import tensorflow.keras as K


def save_config(network, filename):
    """
        network -- model whose configuration should be saved
        filename -- path of the file that the configuration should be saved to
        Returns: None
    """
    config = network.to_json()
    with open(filename, "w") as f:
        f.write(config)


def load_config(filename):
    """
        filename -- path of the file containing the model's configuration in
        JSON format
        Returns: the loaded model
    """
    with open(filename, "r") as f:
        config = f.read()

    return K.models.model_from_json(config)
