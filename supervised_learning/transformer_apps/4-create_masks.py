#|/usr/bin/env python3
"""
Create Masks
"""


import tensorflow.compat.v2 as tf


def create_masks(inputs, target):
    encoder_mask = tf.cast(tf.math.equal(inputs, 0), tf.float32)
    encoder_mask = encoder_mask[:, tf.newaxis, tf.newaxis, :]

    decoder_mask = tf.cast(tf.math.equal(target, 0), tf.float32)
    decoder_mask = decoder_mask[:, tf.newaxis, tf.newaxis, :]

    seq_len_out = tf.shape(target)[1]
    look_ahead_mask = 1 - tf.linalg.band_part(
        tf.ones((seq_len_out, seq_len_out)), -1, 0
    )
    combined_mask = tf.maximum(decoder_mask, look_ahead_mask)
    combined_mask = combined_mask[:, tf.newaxis, :, :]

    return encoder_mask, combined_mask, decoder_mask
