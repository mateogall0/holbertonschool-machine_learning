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
        # Load the saved graph and restore the variables
        saver = tf.train.import_meta_graph(save_path + '.meta')
        saver.restore(sess, save_path)

        # Get the required tensors from the graph's collection
        graph = tf.get_default_graph()
        y_pred = graph.get_tensor_by_name('y_pred:0')
        loss = graph.get_collection('loss')[0]
        accuracy = graph.get_collection('accuracy')[0]

        # Evaluate the network's prediction, accuracy, and loss
        feed_dict = {graph.get_tensor_by_name('X:0'): X, graph.get_tensor_by_name('Y:0'): Y}
        y_pred_val, acc_val, loss_val = sess.run([y_pred, accuracy, loss], feed_dict=feed_dict)

    return y_pred_val, acc_val, loss_val
