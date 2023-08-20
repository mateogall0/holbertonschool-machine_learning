#!/usr/bin/env python3
"""
UNI BLEU
"""


import numpy as np


def uni_bleu(references, sentence):
    """
    Calculates the unigram BLEU score for a sentence
    """
    unique = list(sentence)
    words = {}
    for r in references:
        for word in r:
            if word in unique:
                if word not in words.keys():
                    words[word] = r.count(word)
                else:
                    actual = r.count(word)
                    prev = words[word]
                    words[word] = max(actual, prev)

    c = len(sentence)
    prob = sum(words.values()) / c

    best_match = []
    for r in references:
        ref_len = len(r)
        diff = abs(ref_len - c)
        best_match.append((diff, ref_len))

    sort = sorted(best_match, key=(lambda x: x[0]))
    best = sort[0][1]
    if c > best:
        return np.exp(np.log(prob))
    return np.exp(1 - (best / c)) * np.exp(np.log(prob))
