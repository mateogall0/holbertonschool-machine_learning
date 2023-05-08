#!/usr/bin/en python3
"""
    Sequential
"""


import tensorflow.keras as K
from tensorflow.keras import regularizers
import tensorflow.keras.layers as layers


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
        Builds a neural network with the Keras library

        nx -- number of input features to the network
        layers -- list containing the number of nodes in each layer of the
        network
        activations -- list containing the activation functions used for each
        layer of the network
        lambtha -- L2 regularization parameter
        keep_prob -- probability that a node will be kept for dropout

        Returns: the keras model
    """
    # Create a sequential model
    model = K.Sequential()

    # Add the first hidden layer with L2 regularization
    model.add(layers.Dense(
        layers[0],
        input_shape=(nx,),
        activation=activations[0],
        kernel_regularizer=regularizers.l2(lambtha)
    ))
    model.add(layers.Dropout(1 - keep_prob))

    # Add the remaining hidden layers with L2 regularization and dropout
    for i in range(1, len(layers)):
        model.add(layers.Dense(layers[i],
                                activation=activations[i],
                                kernel_regularizer=regularizers.l2(lambtha)
                                ))
        model.add(layers.Dropout(1 - keep_prob))

    # Add the output layer
    model.add(layers.Dense(1, activation=None))
    return model
