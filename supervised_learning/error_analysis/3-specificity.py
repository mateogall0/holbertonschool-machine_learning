#!/usr/bin/env python3
"""
    Specificity
"""


import numpy as np


def specificity(confusion):
    """
        Calculates the specificity for each class in a confusion matrix

        confusion -- confusion numpy.ndarray of shape (classes, classes)
        where row indices represent the correct labels and column
        indices represent the predicted labels
            classes -- number of classes
        Returns: a numpy.ndarray of shape (classes,) containing the precision
        of each class
    """
    classes = confusion.shape[0]
    specificity = np.zeros((classes,))
    for i in range(classes):
        # sum of true negatives for class i
        trueNegatives = np.sum(confusion[np.arange(classes) != i, :]
                               [:, np.arange(classes) != i])
        # sum of false positives for class i
        falsePositives = np.sum(confusion[:, i]) - confusion[i, i]
        # specificity score for class i
        specificity[i] = trueNegatives / (trueNegatives + falsePositives)
    return specificity
