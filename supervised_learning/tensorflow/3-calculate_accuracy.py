#!/usr/bin/env python3
"""
    Accuracy
"""


import tensorflow as tf


def calculate_accuracy(y, y_pred):
    """
        Calculates the accuracy of a prediction
    """
    # Convert one-hot encoded y to class labels
    y_true = tf.argmax(y, axis=1)
    # Convert one-hot encoded y_pred to class labels
    y_pred = tf.argmax(y_pred, axis=1)
    # Calculate the number of correct predictions
    correct_predictions = tf.equal(y_true, y_pred)
    # Calculate the accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
    return accuracy
