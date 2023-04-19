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
    tf.reset_default_graph()
    x, y = create_placeholders(X_train.shape[1], Y_train.shape[1])
    y_pred = forward_prop(x, layer_sizes, activations)
    loss = calculate_loss(y, y_pred)
    accuracy = calculate_accuracy(y, y_pred)
    train_op = create_train_op(loss, alpha)
    saver = tf.train.Saver()
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        for i in range(iterations+1):
            if i % 100 == 0:
                t_cost, t_acc = sess.run([loss, accuracy], feed_dict={x: X_train, y: Y_train})
                v_cost, v_acc = sess.run([loss, accuracy], feed_dict={x: X_valid, y: Y_valid})
                print("After {} iterations:".format(i))
                print("\tTraining Cost: {}".format(t_cost))
                print("\tTraining Accuracy: {}".format(t_acc))
                print("\tValidation Cost: {}".format(v_cost))
                print("\tValidation Accuracy: {}".format(v_acc))
            if i == 0 or i == iterations:
                t_cost, t_acc = sess.run([loss, accuracy], feed_dict={x: X_train, y: Y_train})
                v_cost, v_acc = sess.run([loss, accuracy], feed_dict={x: X_valid, y: Y_valid})
                print("After {} iterations:".format(i))
                print("\tTraining Cost: {}".format(t_cost))
                print("\tTraining Accuracy: {}".format(t_acc))
                print("\tValidation Cost: {}".format(v_cost))
                print("\tValidation Accuracy: {}".format(v_acc))
            sess.run(train_op, feed_dict={x: X_train, y: Y_train})
        saver.save(sess, save_path)
    return save_path
