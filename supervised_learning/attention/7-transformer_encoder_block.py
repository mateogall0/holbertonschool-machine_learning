#!/usr/bin/env python3
"""
Transformer Encoder Block
"""


import tensorflow as tf
MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention


class EncoderBlock(tf.keras.layers.Layer):
    """
    Encoder block
    """
    def __init__(self, dm, h, hidden, drop_rate=0.1):
        super(EncoderBlock, self).__init__()
        self.mha = MultiHeadAttention(dm, h)
        self.dense_hidden = tf.keras.layers.Dense(hidden, activation='relu')
        self.dense_output = tf.keras.layers.Dense(dm)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate=drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(rate=drop_rate)

    def call(self, x, training, mask=None):
        """
        Call
        """
        attention, _ = self.mha(x, x, x, mask)
        dropout = self.dropout1(attention, training=training)
        attention_x_norm = self.layernorm1(x + dropout)

        hidden = self.dense_hidden(attention_x_norm)
        output = self.dense_output(hidden)

        output = self.dropout2(output)
        return self.layernorm2(attention_x_norm + output)
