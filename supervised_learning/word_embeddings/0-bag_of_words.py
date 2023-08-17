#!/usr/bin/env python3
"""
Bag Of Words
"""


import numpy as np


def bag_of_words(sentences, vocab=None):
    """
    Creates a bag of words embedding matrix
    """
    if vocab is None:
        words = [word for sentence in sentences for word in sentence.split()]
        vocab = list(set(words))
    word_to_index = {word: i for i, word in enumerate(vocab)}

    s = len(sentences)
    f = len(vocab)
    embeddings = np.zeros((s, f), dtype=int)

    for i, sentence in enumerate(sentences):
        words = sentence.split()
        for word in words:
            if word in word_to_index:
                embeddings[i, word_to_index[word]] = 1
    return embeddings, vocab
