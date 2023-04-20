#!/usr/bin/env python3
"""
    Evaluate
"""


import tensorflow as tf


def evaluate(X, Y, save_path):
    """
        Evaluates the output of a neural network
    """
    graph = tf.Graph()
    with tf.compat.v1.Session(graph=graph) as sess:
        saver = tf.compat.v1.train.import_meta_graph(save_path + '.meta')
        saver.restore(sess, save_path)

        # Get the tensors we need
        y_pred = graph.get_collection('y_pred')[0]
        loss = graph.get_collection('loss')[0]
        accuracy = graph.get_collection('accuracy')[0]
        X_placeholder = graph.get_collection('X')[0]
        Y_placeholder = graph.get_collection('Y')[0]

        # Evaluate the model
        feed_dict = {X_placeholder: X, Y_placeholder: Y}
        y_pred_val, loss_val, accuracy_val = sess.run([y_pred, loss, accuracy], feed_dict=feed_dict)

    return y_pred_val, accuracy_val, loss_val
