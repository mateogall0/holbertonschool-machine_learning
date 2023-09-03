#!/usr/bin/env python3
"""
RNN Decoder
"""


import tensorflow as tf
SelfAttention = __import__('1-self_attention').SelfAttention


class RNNDecoder(tf.keras.layers.Layer):
    """
    Decode for machine translation
    """
    def __init__(self, vocab, embedding, units, batch):
        super(RNNDecoder, self).__init__()
        self.embedding = tf.keras.layers.Embedding(
            input_dim=vocab, output_dim=embedding)
        self.gru = tf.keras.layers.GRU(units=units,
                                       recurrent_initializer='glorot_uniform',
                                       return_sequences=True,
                                       return_state=True)
        self.F = tf.keras.layers.Dense(vocab)
        self.attention = SelfAttention(units)

    def call(self, x, s_prev, hidden_states):
        """
        Call
        """
        x = self.embedding(x)
        context, _ = self.attention(s_prev, hidden_states)
        context = tf.expand_dims(context, 1)
        x = tf.concat([context, x], axis=-1)
        outputs, state = self.gru(x, initial_state=s_prev)
        y = self.F(outputs)
        return y, state
