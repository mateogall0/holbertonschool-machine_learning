#!/usr/bin/env python3
"""
    F1 score
"""


sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """
        Calculates the specificity for each class in a confusion matrix

        confusion -- confusion numpy.ndarray of shape (classes, classes)
        where row indices represent the correct labels and column
        indices represent the predicted labels
            classes -- number of classes
        Returns: a numpy.ndarray of shape (classes,) containing the F1 score
        of each class

        F1 = 2 * (precision * recall) / (precision + recall)
    """
    recall = sensitivity(confusion)
    prec = precision(confusion)

    return 2 * (prec * recall) / (prec + recall)
