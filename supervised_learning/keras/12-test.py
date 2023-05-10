#!/usr/bin/env python3
"""
    Test
"""


import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    """
        Tests a neural network

        network -- network model to test
        data -- input data to test the model with
        labels -- correct one-hot labels of data
        verbose -- boolean that determines if output should be printed during
        the testing process
        Returns: the loss and accuracy of the model with the testing data,
        respectively
    """
    return network.evaluate(x=data, y=labels, verbose=verbose)
