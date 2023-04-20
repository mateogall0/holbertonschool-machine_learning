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
    with graph.as_default():
        saver = tf.train.import_meta_graph(save_path + '.meta')

    # Get the tensors we need from the graph's collection
    with tf.Session(graph=graph) as sess:
        saver.restore(sess, save_path)
        y_pred = graph.get_collection('y_pred')[0]
        loss = graph.get_collection('loss')[0]
        accuracy = graph.get_collection('accuracy')[0]

        # Evaluate the model on the input data
        feed_dict = {'X:0': X, 'Y:0': Y}
        y_pred_val, loss_val, accuracy_val = sess.run([y_pred, loss, accuracy], feed_dict=feed_dict)

    return y_pred_val, accuracy_val, loss_val
