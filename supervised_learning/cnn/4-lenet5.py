#!/usr/bin/env python3
"""
LeNet-5 (TensorFlow)
"""


import tensorflow as tf


def lenet5(x, y):
    """
    Builds a modified version of the LeNet-5 architecture using tensorflow

    x -- tf.placeholder of shape (m, 28, 28, 1) containing the input images
    for the network
        m -- number of images
    y -- tf.placeholder of shape (m, 10) containing the one-hot labels for the
    network
    The model should consist of the following layers in order:
        Convolutional layer with 6 kernels of shape 5x5 with same padding
        Max pooling layer with kernels of shape 2x2 with 2x2 strides
        Convolutional layer with 16 kernels of shape 5x5 with valid padding
        Max pooling layer with kernels of shape 2x2 with 2x2 strides
        Fully connected layer with 120 nodes
        Fully connected layer with 84 nodes
        Fully connected softmax output layer with 10 nodes
    All layers requiring initialization should initialize their kernels with
    the he_normal initialization method: 
    f.contrib.layers.variance_scaling_initializer()
    All hidden layers requiring activation should use the relu activation
    function
    Returns:
        a tensor for the softmax activated output
        a training operation that utilizes Adam optimization (with default
        hyperparameters)
        a tensor for the loss of the netowrk
        a tensor for the accuracy of the network
    """
    # Convolutional layer 1
    conv1 = tf.layers.conv2d(inputs=x, filters=6, kernel_size=(5, 5), padding='same',
                             activation=tf.nn.relu,
                             kernel_initializer=tf.contrib.layers.variance_scaling_initializer())

    # Max pooling layer 1
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=(2, 2), strides=(2, 2))

    # Convolutional layer 2
    conv2 = tf.layers.conv2d(inputs=pool1, filters=16, kernel_size=(5, 5), padding='valid',
                             activation=tf.nn.relu,
                             kernel_initializer=tf.contrib.layers.variance_scaling_initializer())

    # Max pooling layer 2
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=(2, 2), strides=(2, 2))

    # Flatten the pool2 output
    flatten = tf.layers.flatten(pool2)

    # Fully connected layer 1
    fc1 = tf.layers.dense(inputs=flatten, units=120, activation=tf.nn.relu,
                          kernel_initializer=tf.contrib.layers.variance_scaling_initializer())

    # Fully connected layer 2
    fc2 = tf.layers.dense(inputs=fc1, units=84, activation=tf.nn.relu,
                          kernel_initializer=tf.contrib.layers.variance_scaling_initializer())

    # Output layer
    logits = tf.layers.dense(inputs=fc2, units=10,
                             kernel_initializer=tf.contrib.layers.variance_scaling_initializer())

    # Softmax activated output
    output = tf.nn.softmax(logits)

    # Loss
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits))

    # Accuracy
    correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Training operation
    optimizer = tf.train.AdamOptimizer()
    train_op = optimizer.minimize(loss)

    return output, train_op, loss, accuracy
