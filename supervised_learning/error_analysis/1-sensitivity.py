#!/usr/bin/env python3
"""
    Sensitibity
"""


import numpy as np


def sensitivity(confusion):
    """
        Calculates the sensitivity for each class in a confusion matrix

        confusion -- confusion numpy.ndarray of shape
        (classes, classes) where row indices represent the correct
        labels and column indices represent the predicted labels
        classes -- number of classes
        Returns: a numpy.ndarray of shape (classes,) containing the
        sensitivity of each class

        To calculate the sensitivity for each class, you need to divide
        the true positives (correctly predicted positives) by the
        total actual positives.
        In other words, you need to divide the value in each element on the
        diagonal of the confusion matrix by the sum of the corresponding row.
    """
    classes = confusion.shape[0]
    sensitivity = np.zeros((classes,))
    for i, item in enumerate(confusion):
        truePositives = confusion[i, i]
        falseNegatives = np.sum(confusion[i, :]) - truePositives
        sensitivity[i] = truePositives / (truePositives + falseNegatives)
    return sensitivity
