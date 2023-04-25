#!/usr/bin/env python3
"""
    Mini-Batch
"""


import tensorflow as tf
shuffle_data = __import__('2-shuffle_data').shuffle_data



def train_mini_batch(X_train, Y_train, X_valid, Y_valid,
                     batch_size=32, epochs=5,
                     load_path="/tmp/model.ckpt",
                     save_path="/tmp/model.ckpt"):
    """
        Ttrains a loaded neural network model
        using mini-batch gradient descent
    """
    # Load the model graph and restore the session
    saver = tf.train.import_meta_graph(load_path + '.meta')
    x = tf.get_collection('x')[0]
    y = tf.get_collection('y')[0]
    accuracy = tf.get_collection('accuracy')[0]
    loss = tf.get_collection('loss')[0]
    train_op = tf.get_collection('train_op')[0]

    # Start a TensorFlow session to run the training operations
    with tf.Session() as sess:
        m = X_train.shape[0]

        # Calculate the number of batches
        if m % batch_size == 0:
            n_batches = m // batch_size
        else:
            n_batches = m // batch_size + 1

        # Train the model for the given number of epochs
        for i in range(epochs + 1):
            # Calculate the loss and acuracy for the training set
            cost_train = sess.run(loss, feed_dict={x: X_train, y: Y_train})
            accuracy_train = sess.run(accuracy, feed_dict={x: X_train, y: Y_train})
            cost_val = sess.run(loss, feed_dict={x: X_valid, y: Y_valid})

            # Calculate the loss and accuracy for the validation set
            accuracy_val = sess.run(accuracy, feed_dict={x: X_valid, y: Y_valid})
            # Print the training and validation results for the current epoch
            print("After {} epochs:".format(i))
            print("\tTraining Cost: {}".format(cost_train))
            print("\tTraining Accuracy: {}".format(accuracy_train))
            print("\tValidation Cost: {}".format(cost_val))
            print("\tValidation Accuracy: {}".format(accuracy_val))

            # Train the model using mini-batches
            if i < epochs:
                # Shuffle the training data
                shuffled_X, shuffled_Y = shuffle_data(X_train, Y_train)

                # Train the model using mini-batches
                for b in range(n_batches):
                    start = b * batch_size
                    end = (b + 1) * batch_size
                    if end > m:
                        end = m
                    X_mini_batch = shuffled_X[start:end]
                    Y_mini_batch = shuffled_Y[start:end]

                    # Define the next mini-batch
                    next_train = {x: X_mini_batch, y: Y_mini_batch}

                    # Run a training step with the mini-batch
                    sess.run(train_op, feed_dict=next_train)

                    # Print the mini-batch results every 100 batches
                    if (b + 1) % 100 == 0 and b != 0:
                        loss_mini_batch = sess.run(loss, feed_dict=next_train)
                        acc_mini_batch = sess.run(accuracy,
                                                  feed_dict=next_train)
                        print("\tStep {}:".format(b + 1))
                        print("\t\tCost: {}".format(loss_mini_batch))
                        print("\t\tAccuracy: {}".format(acc_mini_batch))
        # Save the trained model
        return saver.save(sess, save_path)
