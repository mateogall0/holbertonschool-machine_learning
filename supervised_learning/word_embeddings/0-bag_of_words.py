#!/usr/bin/env python3
"""
Bag Of Words
"""


from sklearn.feature_extraction.text import CountVectorizer


def bag_of_words(sentences, vocab=None):
    """
    Creates a bag of words embedding matrix
    """
    v = CountVectorizer(vocabulary=vocab)
    X = v.fit_transform(sentences)
    return X.toarray(), v.get_feature_names()
