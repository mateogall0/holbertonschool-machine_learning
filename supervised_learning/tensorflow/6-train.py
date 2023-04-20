#!/usr/bin/env python3
"""
    Train
"""


import tensorflow as tf


calculate_accuracy = __import__('3-calculate_accuracy').calculate_accuracy
calculate_loss = __import__('4-calculate_loss').calculate_loss
create_placeholders = __import__('0-create_placeholders').create_placeholders
create_train_op = __import__('5-create_train_op').create_train_op
forward_prop = __import__('2-forward_prop').forward_prop


def train(X_train, Y_train, X_valid, Y_valid, layer_sizes,
          activations, alpha, iterations, save_path="/tmp/model.ckpt"):
    """
    Builds, trains, and saves a neural network classifier.
    """
    # Initialize TensorFlow's random seed to a fixed value.
    tf.set_random_seed(0)

    # Get the number of training examples (m) and the
    # number of features (nx) from the training input data.

    m, nx = X_train.shape
    # Get the number of classes (classes) from the training labels data.

    classes = Y_train.shape[1]

    # Create TensorFlow placeholders (x and y) for the input data and labels.
    x, y = create_placeholders(nx, classes)

    # Build the neural network by calling the forward_prop function with
    # the specified layer sizes and activation functions.
    y_pred = forward_prop(x, layer_sizes, activations)

    # Calculate the loss and accuracy of the neural network using the
    # calculate_loss and calculate_accuracy functions, respectively.
    loss = calculate_loss(y, y_pred)
    accuracy = calculate_accuracy(y, y_pred)

    # Create the training operation using the create_train_op function
    # with the calculated loss and the specified learning rate.
    train_op = create_train_op(loss, alpha)

    # Create a TensorFlow session and initialize all variables.
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)

        # Add the input data placeholder (x), label placeholder (y),
        # predicted labels tensor (y_pred), loss tensor, accuracy tensor, and
        # training operation to the graph's collection for easy access later.
        tf.add_to_collection('x', x)
        tf.add_to_collection('y', y)
        tf.add_to_collection('y_pred', y_pred)
        tf.add_to_collection('loss', loss)
        tf.add_to_collection('accuracy', accuracy)
        tf.add_to_collection('train_op', train_op)

        # Train the neural network for the specified number of iterations.
        # After every 100 iterations, and at the 0th iteration and the last
        # iteration, print the training and validation cost and accuracy.
        for i in range(iterations + 1):
            cost_train, acc_train = sess.run(
                [loss, accuracy], feed_dict={x: X_train, y: Y_train})
            cost_valid, acc_valid = sess.run(
                [loss, accuracy], feed_dict={x: X_valid, y: Y_valid})
            if i % 100 == 0 or i == 0 or i == iterations:
                print("After {} iterations:".format(i))
                print("\tTraining Cost: {}".format(cost_train))
                print("\tTraining Accuracy: {}".format(acc_train))
                print("\tValidation Cost: {}".format(cost_valid))
                print("\tValidation Accuracy: {}".format(acc_valid))

            if i < iterations:
                sess.run(train_op, feed_dict={x: X_train, y: Y_train})

        # Save the trained model to the specified path
        # using a TensorFlow Saver object.
        saver = tf.train.Saver()
        save_path = saver.save(sess, save_path)
    # Return the path where the model was saved.
    return save_path
