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
    
    with tf.Session() as sess:
        # Restore the trained model
        saver.restore(sess, load_path)
        
        # Loop over epochs
        for epoch in range(epochs + 1):
            
            # Shuffle the training data
            X_train, Y_train = shuffle_data(X_train, Y_train)
            
            # Calculate training and validation cost and accuracy after each epoch
            train_cost, train_accuracy = sess.run([loss, accuracy], feed_dict={x: X_train, y: Y_train})
            valid_cost, valid_accuracy = sess.run([loss, accuracy], feed_dict={x: X_valid, y: Y_valid})
            
            # Print progress after each epoch
            print("After {} epochs:".format(epoch))
            print("\tTraining Cost: {}\n\tTraining Accuracy: {}".format(train_cost, train_accuracy))
            print("\tValidation Cost: {}\n\tValidation Accuracy: {}".format(valid_cost, valid_accuracy))

            # Loop over batches
            for i in range(0, X_train.shape[0], batch_size):
                # Get the next batch of data
                X_batch = X_train[i:i+batch_size]
                Y_batch = Y_train[i:i+batch_size]

                # Train the model on the current batch
                _, step_cost, step_accuracy = sess.run([train_op, loss, accuracy], feed_dict={x: X_batch, y: Y_batch})
                
                # Print training progress every 100 steps
                if i != 0 and i % (100 * batch_size) == 0:
                    print("\tStep {}:\n\t\tCost: {}\n\t\tAccuracy: {}".format(i, step_cost, step_accuracy))

        # Save the trained model
        save_path = saver.save(sess, save_path)
        
    return save_path
