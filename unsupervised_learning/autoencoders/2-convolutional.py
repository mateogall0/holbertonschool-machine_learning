#!/usr/bin/env python3
"""
Convolutional
"""


import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """
    Creates a convolutional autoencoder
    """
    encoder_inputs = keras.Input(shape=input_dims)
    x = encoder_inputs
    for f in filters:
        x = keras.layers.Conv2D(f, (3, 3), activation='relu',
                                padding='same')(x)
        x = keras.layers.MaxPooling2D((2, 2), padding='same')(x)
    encoder_outputs = x
    encoder = keras.models.Model(encoder_inputs, encoder_outputs,
                                 name='encoder')

    latent_inputs = keras.Input(shape=latent_dims)
    x = latent_inputs
    for f in reversed(filters[:-1]):
        x = keras.layers.Conv2D(f, (3, 3), activation='relu',
                                padding='same')(x)
        x = keras.layers.UpSampling2D((2, 2))(x)
    x = keras.layers.Conv2D(filters[-1], (3, 3), activation='relu',
                            padding='valid')(x)
    x = keras.layers.UpSampling2D((2, 2))(x)
    decoder_outputs = keras.layers.Conv2D(
        input_dims[-1], (3, 3), activation='sigmoid', padding='same'
    )(x)
    decoder = keras.models.Model(latent_inputs, decoder_outputs,
                                 name='decoder')

    autoencoder_outputs = decoder(encoder(encoder_inputs))
    autoencoder = keras.models.Model(
        encoder_inputs, autoencoder_outputs, name='autoencoder'
    )

    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    return encoder, decoder, autoencoder
