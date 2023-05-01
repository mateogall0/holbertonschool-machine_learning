#!/usr/bin/env python3
"""
    Create Confusion
"""


import numpy as np


def create_confusion_matrix(labels, logits):
    """
        Creates a confusion matrix

        labels -- one-hot numpy.ndarray of shape (m, classes) containing
        the correct labels for each data point
        m -- number of data points
        classes -- number of classes
        logits -- one-hot numpy.ndarray of shape (m, classes) containing
        the predicted labels
        Returns: confusion numpy.ndarray of shape (classes, classes) wit
    """
    m, classes = labels.shape

    # Create a zero-filled confusion matrix
    confusion = np.zeros((classes, classes))

    # Convert one-hot
    correct_labels = np.argmax(labels, axis=1)
    predicted_labels = np.argmax(logits, axis=1)

    # For each data point, increment the corresponding cell of
    # the confusion matrix
    for i in range(m):
        confusion[correct_labels[i], predicted_labels[i]] += 1
    return confusion
