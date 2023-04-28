#!/usr/bin/env python3
"""
    Put it all together and what do you get?
"""


import tensorflow as tf


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
    X_train, y_train = Data_train
    X_valid, y_valid = Data_valid
    input_shape = X_train.shape[1:]
    num_classes = y_train.shape[1]

    # Build model
    X = tf.placeholder(tf.float32, shape=(None, *input_shape), name='X')
    y = tf.placeholder(tf.float32, shape=(None, num_classes), name='y')
    is_training = tf.placeholder(tf.bool, name='is_training')

    x = X
    for layer, activation in zip(layers, activations):
        x = tf.layers.dense(x, layer, activation=None)
        x = tf.layers.batch_normalization(x, training=is_training)
        x = activation(x)

    logits = tf.layers.dense(x, num_classes, activation=None)
    y_proba = tf.nn.softmax(logits, name='y_proba')

    # Define loss and optimizer
    with tf.name_scope('loss'):
        xentropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=logits)
        cost = tf.reduce_mean(xentropy, name='cost')
    with tf.name_scope('train'):
        global_step = tf.Variable(0, trainable=False, name='global_step')
        decay_steps = X_train.shape[0] // batch_size
        learning_rate = tf.train.inverse_time_decay(alpha, global_step, decay_steps, decay_rate)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1, beta2=beta2, epsilon=epsilon)
        train_op = optimizer.minimize(cost, global_step=global_step)

    # Define accuracy
    with tf.name_scope('accuracy'):
        correct = tf.equal(tf.argmax(y_proba, axis=1), tf.argmax(y, axis=1))
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name='accuracy')

    # Initialize variables and start session
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        init.run()

        # Train model
        for epoch in range(1, epochs+1):
            print("Epoch {}:".format(epoch))
            total_step_cost = 0
            total_step_accuracy = 0

            # Shuffle data
            shuffled_indices = np.random.permutation(X_train.shape[0])
            X_train_shuffled = X_train[shuffled_indices]
            y_train_shuffled = y_train[shuffled_indices]

            # Train on mini-batches
            for batch in range(0, X_train.shape[0], batch_size):
                batch_X = X_train_shuffled[batch:batch+batch_size]
                batch_y = y_train_shuffled[batch:batch+batch_size]
                _, step, step_cost, step_accuracy = sess.run([train_op, global_step, cost, accuracy], feed_dict={X: batch_X, y: batch_y, is_training: True})
                total_step_cost += step_cost
                total_step_accuracy += step_accuracy

                # Print progress every 100 steps
                if step % 100 == 0:
                    avg_step_cost = total_step_cost / 100
                    avg_step_accuracy = total_step_accuracy / 100
                    print("\tStep {}:".format(step))
                    print("\t\tCost: {}".format(avg_step_accuracy))
                    print("\t\tAccuracy: {}".format(step_accuracy))
            # Compute cost and accuracy on entire training set
                train_cost, train_accuracy = sess.run([cost, accuracy], feed_dict={X: X_train, y: y_train})
                
                # Compute cost and accuracy on entire validation set
                valid_cost, valid_accuracy = sess.run([cost, accuracy], feed_dict={X: X_valid, y: y_valid})
                
                print(f"After {epoch+1} epochs:")
                print(f"\tTraining Cost: {train_cost}")
                print(f"\tTraining Accuracy: {train_accuracy}")
                print(f"\tValidation Cost: {valid_cost}")
                print(f"\tValidation Accuracy: {valid_accuracy}")
                
                # Save model after every epoch
                saver.save(sess, save_path)
                
        return save_path
