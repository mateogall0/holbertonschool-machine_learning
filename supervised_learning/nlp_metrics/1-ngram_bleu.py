#!/usr/bin/env python3
"""
N-gram BLEU score
"""


import numpy as np


def n_gram(sentence, n):
    """
    Sentence to grams
    """
    if n <= 1:
        return sentence
    step = n - 1
    result = sentence[:-step]
    for i, _ in enumerate(result):
        for j in range(step):
            result[i] += ' ' + sentence[i + 1 + j]
    return result


def ngram_bleu(references, sentence, n):
    """
    Calculates the n-gram BLEU score for a sentence
    """
    c = len(sentence)
    rs = [len(r) for r in references]
    sentence = n_gram(sentence, n)
    references = list(map(lambda ref: n_gram(ref, n), references))
    flat = set([g for ref in references for g in ref])

    top = 0
    for g in flat:
        if g in sentence:
            top += 1
    precision = top / len(sentence)
    best = None
    for i, item in enumerate(references):
        if best is None:
            best = item
            ri = i
            continue
        if abs(len(item) - len(sentence)) < abs(len(best) - len(sentence)):
            best = item
            ri = i
    r = rs[ri]
    if c > r:
        penalty = 1
    else:
        penalty = np.exp(1 - r / c)
    return penalty * precision
