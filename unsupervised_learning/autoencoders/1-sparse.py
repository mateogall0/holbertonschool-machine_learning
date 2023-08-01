#!/usr/bin/env python3
"""
Sparse
"""


import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims, lambtha):
    """
    Creates a sparse autoencoder
    """
    encoder_inputs = keras.Input(shape=(input_dims,))
    x = encoder_inputs
    for nodes in hidden_layers:
        x = keras.layers.Dense(nodes, activation='relu')(x)
    encoder_outputs = keras.layers.Dense(
        latent_dims, activation='relu',
        activity_regularizer=keras.regularizers.l1(lambtha)
    )(x)
    encoder = keras.models.Model(
        encoder_inputs, encoder_outputs, name='encoder'
    )

    decoder_inputs = keras.Input(shape=(latent_dims,))
    x = decoder_inputs
    for nodes in reversed(hidden_layers):
        x = keras.layers.Dense(nodes, activation='relu')(x)
    decoder_outputs = keras.layers.Dense(input_dims, activation='sigmoid')(x)
    decoder = keras.models.Model(
        decoder_inputs, decoder_outputs, name='decoder'
    )

    autoencoder_outputs = decoder(encoder(encoder_inputs))
    autoencoder = keras.models.Model(
        encoder_inputs, autoencoder_outputs, name='autoencoder'
    )

    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    return encoder, decoder, autoencoder
