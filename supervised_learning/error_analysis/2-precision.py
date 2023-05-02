#!/usr/bin/env python3
"""
    Precision
"""


import numpy as np


def precision(confusion):
    """
        Calculates the precision for each class in a confusion matrix

        confusion -- confusion numpy.ndarray of shape (classes, classes)
        where row indices represent the correct labels and column
        indices represent the predicted labels
            classes -- number of classes
        Returns: a numpy.ndarray of shape (classes,) containing the precision
        of each class

        Difference with sensitivity is in:
            Sensitivity: confusion[i, :]
            Precision: confusion[:, i]
    """
    classes = confusion.shape[0]
    precision = np.zeros((classes,))
    for i, item in enumerate(confusion):
        truePositives = confusion[i, i]
        falseNegatives = np.sum(confusion[:, i]) - truePositives
        precision[i] = truePositives / (truePositives + falseNegatives)
    return precision
