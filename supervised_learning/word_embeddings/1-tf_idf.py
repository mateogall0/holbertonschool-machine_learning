#!/usr/bin/env python3
"""
TF-IDF
"""


from sklearn.feature_extraction.text import TfidfVectorizer


def tf_idf(sentences, vocab=None):
    """
    Creates a TF-IDF embedding
    """
    vectorizer = TfidfVectorizer(vocabulary=vocab)
    return (
        vectorizer.fit_transform(sentences).toarray(),
        vectorizer.get_feature_names()
    )
