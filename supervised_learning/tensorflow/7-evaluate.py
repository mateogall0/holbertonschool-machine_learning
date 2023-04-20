#!/usr/bin/env python3
"""
    Evaluate
"""


import tensorflow as tf


def evaluate(X, Y, save_path):
    """
        Evaluates the output of a neural network
    """
    with tf.Session() as sess:
        # Load the saved model
        saver = tf.train.import_meta_graph(save_path + '.meta')
        saver.restore(sess, save_path)

        # Get tensors from the graph's collection
        y_pred = tf.get_collection('y_pred')[0]
        loss = tf.get_collection('loss')[0]
        accuracy = tf.get_collection('accuracy')[0]

        # Evaluate the model
        feed_dict = {'X:0': X, 'Y:0': Y}
        y_pred_val, loss_val, acc_val = sess.run([y_pred, loss, accuracy], feed_dict=feed_dict)

    return y_pred_val, acc_val, loss_val
