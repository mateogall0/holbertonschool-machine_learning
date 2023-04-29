#!/usr/bin/env python3
"""
    Put it all together and what do you get?
"""


import tensorflow as tf
import numpy as np


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


def shuffle_data(X, Y):
    """
        Shuffles the data points in two matrices the same way
    """
    m = X.shape[0]

    # Generate a random permutation of integers from 0 to m-1
    permutation = np.random.permutation(m)

    # Shuffle the rows of X and Y using the permutation
    shuffled_X = X[permutation]
    shuffled_Y = Y[permutation]

    # Return the shuffled X and Y matrices as a tuple
    return shuffled_X, shuffled_Y


def create_placeholders(nx, classes):
    """
        Returns two placeholders
    """
    x = tf.placeholder(tf.float32, shape=(None, nx), name="x")
    y = tf.placeholder(tf.float32, shape=(None, classes), name="y")
    return x, y


def create_layer(prev, n, activation):
    """
        Returns: the tensor output of the layer
    """
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.Dense(units=n, activation=activation,
                            kernel_initializer=init, name="layer")
    return layer(prev)


def forward_prop(x, layer_sizes=[], activations=[]):
    """
        Creates the forward propagation graph
    """
    prev = x
    for i in range(len(layer_sizes)):
        with tf.variable_scope("layer" + str(i+1), reuse=tf.AUTO_REUSE):
            layer = create_layer(prev, layer_sizes[i], activations[i])
            prev = layer
    return layer


def calculate_loss(y, y_pred):
    """
        Calculates the softmax cross-entropy loss of a prediction
    """
    return tf.losses.softmax_cross_entropy(onehot_labels=y, logits=y_pred)


def calculate_accuracy(y, y_pred):
    """
        Calculates the accuracy of a prediction
    """
    # Convert one-hot encoded y to class labels
    y_true = tf.argmax(y, axis=1)
    # Convert one-hot encoded y_pred to class labels
    y_pred = tf.argmax(y_pred, axis=1)
    # Calculate the number of correct predictions
    correct_predictions = tf.equal(y_true, y_pred)
    # Calculate the accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
    return accuracy


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
    nx = Data_train[0].shape[1]
    classes = Data_train[1].shape[1]

    (X_train, Y_train) = Data_train
    (X_valid, Y_valid) = Data_valid

    x = tf.placeholder(tf.float32, shape=[None, nx], name='x')
    y = tf.placeholder(tf.float32, shape=[None, classes], name='y')

    tf.add_to_collection('x', x)
    tf.add_to_collection('y', y)

    y_pred = forward_prop(x, layers, activations)
    tf.add_to_collection('y_pred', y_pred)

    loss = calculate_loss(y, y_pred)
    tf.add_to_collection('loss', loss)

    accuracy = calculate_accuracy(y, y_pred)
    tf.add_to_collection('accuracy', accuracy)

    global_step = tf.Variable(0)
    alpha_op = learning_rate_decay(alpha, decay_rate, global_step, 1)
    train_op = create_Adam_op(loss, alpha_op, beta1, beta2, epsilon)
    tf.add_to_collection('train_op', train_op)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)
        m = X_train.shape[0]
        if (m % batch_size) == 0:
            num_minibatches = int(m / batch_size)
            check = 1
        else:
            num_minibatches = int(m / batch_size) + 1
            check = 0

        for epoch in range(epochs + 1):
            feed_train = {x: X_train, y: Y_train}
            feed_valid = {x: X_valid, y: Y_valid}
            train_cost = sess.run(loss, feed_dict=feed_train)
            train_accuracy = sess.run(accuracy, feed_dict=feed_train)
            valid_cost = sess.run(loss, feed_dict=feed_valid)
            valid_accuracy = sess.run(accuracy, feed_dict=feed_valid)

            print("After {} epochs:".format(epoch))
            print("\tTraining Cost: {}".format(train_cost))
            print("\tTraining Accuracy: {}".format(train_accuracy))
            print("\tValidation Cost: {}".format(valid_cost))
            print("\tValidation Accuracy: {}".format(valid_accuracy))

            if epoch < epochs:
                Xs, Ys = shuffle_data(X_train, Y_train)

                for step_number in range(num_minibatches):
                    start = step_number * batch_size
                    end = (step_number + 1) * batch_size
                    if check == 0 and step_number == num_minibatches - 1:
                        x_minbatch = Xs[start::]
                        y_minbatch = Ys[start::]
                    else:
                        x_minbatch = Xs[start:end]
                        y_minbatch = Ys[start:end]

                    feed_mini = {x: x_minbatch, y: y_minbatch}
                    sess.run(train_op, feed_dict=feed_mini)

                    if ((step_number + 1) % 100 == 0) and (step_number != 0):
                        step_cost = sess.run(loss, feed_dict=feed_mini)
                        step_accuracy = sess.run(accuracy, feed_dict=feed_mini)
                        print("\tStep {}:".format(step_number + 1))
                        print("\t\tCost: {}".format(step_cost))
                        print("\t\tAccuracy: {}".format(step_accuracy))
            sess.run(tf.assign(global_step, global_step + 1))
            save_path = saver.save(sess, save_path)
    return save_path