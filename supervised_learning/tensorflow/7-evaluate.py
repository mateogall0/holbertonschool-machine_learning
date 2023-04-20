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
        sess = tf.Session()
        saver = tf.train.import_meta_graph(save_path + '.meta')
        saver.restore(sess, save_path)

        # Get the tensors we need from the graph's collection
        y_pred = graph.get_tensor_by_name('y_pred:0')
        loss = graph.get_tensor_by_name('loss:0')
        accuracy = graph.get_tensor_by_name('accuracy:0')

        # Evaluate the model on the input data
        feed_dict = {'X:0': X, 'Y:0': Y}
        y_pred_val, loss_val, accuracy_val = sess.run([y_pred, loss, accuracy], feed_dict=feed_dict)

    sess.close()

    return y_pred_val, accuracy_val, loss_val
