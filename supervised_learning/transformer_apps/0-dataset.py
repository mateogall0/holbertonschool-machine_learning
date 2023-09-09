#!/usr/bin/env python3
"""
Dataset
"""


import tensorflow as tf
import tensorflow_datasets as tfds


class Dataset:
    def __init__(self):
        ex, _ = tfds.load("ted_hrlr_translate/pt_to_en",
                             as_supervised=True,
                             with_info=True)
        self.data_train = ex['train']
        self.data_valid = ex['validation']
        self.otkenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train
        )

    def tokenize_dataset(self, data):
        tokenizer_pt = tfds.features.text.SubwordTextEncoder.build_from_corpus(
            (pt.numpy() for pt, _ in data), target_vocab_size=2 ** 15
        )

        tokenizer_en = tfds.features.text.SubwordTextEncoder.build_from_corpus(
            (en.numpy() for _, en in data), target_vocab_size=2 ** 15
        )

        return tokenizer_pt, tokenizer_en
