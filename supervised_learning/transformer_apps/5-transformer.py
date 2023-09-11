#!/usr/bin/env python3
"""
Train
"""


import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds


tf.compat.v1.enable_eager_execution()


class Dataset:
    def __init__(self, batch_size, max_len):
        def filter_max_length(x, y, max_length=max_len):
            return tf.logical_and(tf.size(x) <= max_length,
                                  tf.size(y) <= max_length)

        examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en',
                                       with_info=True,
                                       as_supervised=True)

        self.data_train = examples['train']
        self.data_valid = examples['validation']

        PT, EN = self.tokenize_dataset(self.data_train)
        self.tokenizer_pt, self.tokenizer_en = PT, EN

        self.data_train = self.data_train.map(self.tf_encode)
        self.data_train = self.data_train.filter(filter_max_length)
        self.data_train = self.data_train.cache()

        shu = metadata.splits['train'].num_examples
        self.data_train = self.data_train.shuffle(shu)
        pad_shape = ([None], [None])
        self.data_train = self.data_train.padded_batch(batch_size,
                                                       padded_shapes=pad_shape)
        aux = tf.data.experimental.AUTOTUNE
        self.data_train = self.data_train.prefetch(aux)

        self.data_valid = self.data_valid.map(self.tf_encode)
        self.data_valid = self.data_valid.filter(filter_max_length)
        self.data_valid = self.data_valid.padded_batch(batch_size,
                                                       padded_shapes=pad_shape)

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