#!/usr/bin/env python3
"""
    Put it all together and what do you get?
"""


import tensorflow as tf
import numpy as np


def model(Data_train, Data_valid, layers, activations,
          alpha=0.001, beta1=0.9, beta2=0.999,
          epsilon=1e-8, decay_rate=1, batch_size=32,
          epochs=5, save_path='/tmp/model.ckpt'):
    """
        Builds, trains, and saves a neural network model
        in tensorflow using Adam optimization,
        mini-batch gradient descent, learning rate
        decay, and batch normalization

        Data_train -- tuple containing the training
        inputs and training labels, respectively
        Data_valid -- tuple containing the validation
        inputs and validation labels, respectively
        layers -- list containing the number of nodes
        in each layer of the network
        activation -- list containing the activation
        functions used for each layer of the network
        alpha -- learning rate
        beta1 -- weight for the first moment of Adam
        Optimization
        beta2 -- weight for the second moment of Adam
        Optimization
        epsilon -- small number used to avoid division
        by zero
        decay_rate -- decay rate for inverse time decay
        of the learning rate (the corresponding decay
        step should be 1)
        batch_size -- number of data points that should
        be in a mini-batch
        epochs -- number of times the training should
        pass through the whole dataset
        save_path -- path where the model should be saved to
    """
    pass

def create_momentum_op(loss, alpha, beta1):
    """
        Creates the training operation for a neural
        network in tensorflow using the gradient
        descent with momentum optimization algorithm

        loss -- the loss of the network
        alpha -- the learning rate
        beta1 -- the momentum weight
    """
    optimizer = tf.train.MomentumOptimizer(learning_rate=alpha, momentum=beta1)
    train_op = optimizer.minimize(loss)
    return train_op

def create_RMSProp_op(loss, alpha, beta2, epsilon):
    """
        Creates the training operation for a neural
        network in tensorflow using the
        RMSProp optimization algorithm

        loss -- the loss of the network
        alpha -- learning rate
        beta2 -- RMSProp weight
        epsilon -- small number to avoid division by zero
    """
    optimizer = tf.train.RMSPropOptimizer(learning_rate=alpha,
                                          decay=beta2,
                                          epsilon=epsilon)
    return optimizer.minimize(loss)

def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    """
        Creates the training operation for a
        neural network in tensorflow using the
        Adam optimization algorithm

        loss -- loss of the network
        alpha -- learning rate
        beta1 -- weight used for the first moment
        beta2 -- weight used for the second moment
        epsilon -- small number to avoid division by zero
    """
    # Create Adam optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=alpha,
                                       beta1=beta1,
                                       beta2=beta2,
                                       epsilon=epsilon)
    return optimizer.minimize(loss)

def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
        Creates a learning rate decay operation in tensorflow
        using inverse time decay

        alpha -- original learning rate
        decay_rate -- weight used to determine the rate at which alpha will
        decay
        global_step -- number of passes of gradient descent that have elapsed
        decay_step -- number of passes of gradient descent that
        should occur before alpha is decayed further



        If staircase=True, the learning rate will decay in a stepwise
        fashion, meaning that the learning rate will be reduced by a
        factor of decay_rate every decay_step steps. This is useful
        when you want to apply a larger decay to the learning rate at
        specific intervals, such as at the end of an epoch or after a
        certain number of steps.

        If staircase=False, the learning rate will decay continuously,
        meaning that the learning rate will be decayed by a factor of
        decay_rate for every step, which can be useful when you want a
        more gradual and continuous decay.
    """
    return tf.train.inverse_time_decay(
        alpha,
        global_step=global_step,
        decay_steps=decay_step,
        decay_rate=decay_rate,
        staircase=True
    )

def create_batch_norm_layer(prev, n, activation):
    """
        Creates a batch normalization layer for a
        neural network in tensorflow

        prev -- activated output of the previous layer
        n -- number of nodes in the layer to be created
        activation -- activation function that should
        be used on the output of the layer
    """
    # Layers
    k_init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    output = tf.layers.Dense(units=n, kernel_initializer=k_init)
    Z = output(prev)

    # Gamma and Beta initialization
    gamma = tf.Variable(initial_value=tf.constant(1.0, shape=[n]),
                        name="gamma")
    beta = tf.Variable(initial_value=tf.constant(0.0, shape=[n]), name="beta")

    # Batch normalization
    mean, var = tf.nn.moments(Z, axes=0)
    b_norm = tf.nn.batch_normalization(Z, mean, var, offset=beta,
                                       scale=gamma,
                                       variance_epsilon=1e-8)
    if activation is None:
        return b_norm
    return activation(b_norm)
