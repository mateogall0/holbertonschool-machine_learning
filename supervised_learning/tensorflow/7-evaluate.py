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

    # Extract the tensors we need from the graph's collection
    with tf.Session(graph=graph) as sess:
        # Load the weights from the checkpoint
        saver.restore(sess, save_path)
        
        # Get the input and output tensors
        X_input = graph.get_tensor_by_name('input:0')
        y_pred = graph.get_tensor_by_name('output:0')
        
        # Get the loss tensor
        loss = tf.get_collection('loss')[0]
        
        # Get the accuracy tensor
        accuracy = tf.get_collection('accuracy')[0]

        # Evaluate the network
        feed_dict = {X_input: X, y_true: Y}
        y_pred_val, loss_val, acc_val = sess.run([y_pred, loss, accuracy], feed_dict=feed_dict)

    return y_pred_val, acc_val, loss_val
