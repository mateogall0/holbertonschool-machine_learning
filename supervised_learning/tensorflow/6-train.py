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
	# Set random seed
    tf.set_random_seed(42)
    
    # Get input and output sizes
    input_size = X_train.shape[1]
    output_size = Y_train.shape[1]

    # Create placeholders
    x, y = create_placeholders(input_size, output_size)

    # Forward propagation
    y_pred = forward_prop(x, layer_sizes, activations)

    # Loss function
    loss = calculate_loss(y, y_pred)

    # Accuracy function
    accuracy = calculate_accuracy(y, y_pred)

    # Training operation
    train_op = create_train_op(loss, alpha)

    # Initialize variables
    init = tf.global_variables_initializer()

    # Create session
    with tf.Session() as sess:

        # Initialize variables
        sess.run(init)

        # Train the model
        for i in range(iterations):
            _, train_cost, train_acc = sess.run([train_op, loss, accuracy], feed_dict={x: X_train, y: Y_train})
            valid_cost, valid_acc = sess.run([loss, accuracy], feed_dict={x: X_valid, y: Y_valid})

            if i == 0 or (i + 1) % 100 == 0 or i == iterations - 1:
                print("After {} iterations:".format(i+1))
                print("\tTraining Cost: {}".format(train_cost))
                print("\tTraining Accuracy: {}".format(train_acc))
                print("\tValidation Cost: {}".format(valid_cost))
                print("\tValidation Accuracy: {}".format(valid_acc))

        # Save the model
        saver = tf.train.Saver()
        save_path = saver.save(sess, save_path)
        print("Model saved in file: %s" % save_path)

    return save_path
