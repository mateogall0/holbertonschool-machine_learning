#!/usr/bin/env python3
"""
RNN Encoder
"""


import tensorflow as tf


class RNNEncoder(tf.keras.layers.Layer):
    """
    Encode for machine translation
    """
    def __init__(self, vocab, embedding, units, batch):
        super(RNNEncoder, self).__init__()
        self.batch = batch
        self.units = units
        self.embedding = tf.keras.layers.Embedding(
                input_dim=vocab, output_dim=embedding
            )
        self.gru = tf.keras.layers.GRU(units=self.units,
                                       recurrent_initializer='glorot_uniform',
                                       return_sequences=True,
                                       return_state=True)

    def initialize_hidden_state(self):
        """
        Initializes the hidden states for the RNN cell to a tensor of zeros
        """
        return tf.zeros((self.batch, self.units))

    def call(self, x, initial):
        """
        Call
        """
        x = self.embedding(x)
        outputs, hidden = self.gru(x, initial_state=initial)
        return outputs, hidden
