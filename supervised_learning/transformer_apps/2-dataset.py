#!/usr/bin/env python3
"""
Dataset
"""


import tensorflow.compat.v2 as tf
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

    def encode(self, pt, en):
        portuguese = [self.tokenizer_pt.vocab_size] + self.tokenizer_pt.encode(
            pt.numpy()
        ) + [self.tokenizer_pt.vocab_size + 1]

        english = [self.tokenizer_en.vocab_size] + self.tokenizer_en.encode(
            en.numpy()
        ) + [self.tokenizer_en.vocab_size + 1]

        return portuguese, english

    def tf_encode(self, pt, en):
        def encode_py_function(pt, en):
            pt_encoded, en_encoded = self.encode(pt, en)
            pt_encoded = tf.convert_to_tensor(pt_encoded, dtype=tf.int64)
            en_encoded = tf.convert_to_tensor(en_encoded, dtype=tf.int64)

        pt_encoded, en_encoded = tf.numpy_function(
            encode_py_function, [pt, en], [tf.int64, tf.int64]
        )
        pt_encoded.set_shape([None])
        en_encoded.set_shape([None])

        return pt_encoded, en_encoded
