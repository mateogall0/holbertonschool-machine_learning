#!/usr/bin/env python3
"""
    Sequential
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
    # Create a sequential model
    model = K.Sequential()

    # Add layers to the model
    for i in range(len(layers)):
        if i == 0:
            # Add the first hidden layer with input shape and L2 regularization
            model.add(K.layers.Dense(
                layers[i],
                input_shape=(nx,),
                activation=activations[i],
                kernel_regularizer=K.regularizers.l2(lambtha)
                ))
        else:
            # Add dropout layer and hidden layer with L2 regularization
            model.add(K.layers.Dropout(1 - keep_prob))
            model.add(K.layers.Dense(
                layers[i],
                activation=activations[i],
                kernel_regularizer=K.regularizers.l2(lambtha)
                ))
    return model
