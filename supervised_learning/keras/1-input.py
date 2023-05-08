#!/usr/bin/env python3
"""
    Input
"""


import tensorflow.keras as K


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
    # Initialize the input layer
    inputs = K.Input(shape=(nx,))

    # Initialize the previous layer to be the input layer
    prev_layer = inputs

    # Iterate over the layers, adding each one to the model
    for i in range(len(layers)):
        # Add a dense layer with the specified number of nodes and activation
        # function
        dense_layer = K.layers.Dense(
            layers[i],
            activation=activations[i],
            kernel_regularizer=K.regularizers.l2(lambtha))(prev_layer)

        # Add dropout after the dense layer
        dropout_layer = K.layers.Dropout(1 - keep_prob)(dense_layer)

        # Set the previous layer to be the dropout layer for the next iteration
        prev_layer = dropout_layer

    # Initialize the model with the input and output layers
    model = K.Model(inputs=inputs, outputs=prev_layer)

    return model
