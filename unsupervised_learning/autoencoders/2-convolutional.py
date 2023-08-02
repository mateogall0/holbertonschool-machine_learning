#!/usr/bin/env python3
"""
Convolutional
"""


import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """
    Creates a convolutional autoencoder
    """
    input_img = keras.Input(shape=input_dims)
    x = input_img
    for f in filters:
        x = keras.layers.Conv2D(f, (3, 3), activation='relu',
                                padding='same')(x)
        x = keras.layers.MaxPooling2D((2, 2), padding='same')(x)
    encoder = keras.Model(input_img, x, name='encoder')
    input_decoder = keras.Input(shape=latent_dims)
    decoded = keras.layers.Conv2D(filters[-1],
                                  kernel_size=(3, 3),
                                  padding='same',
                                  activation='relu')(input_decoder)
    decoded = keras.layers.UpSampling2D((2, 2))(decoded)

    for f in range(len(filters) - 2, 0, -1):
        decoded = keras.layers.Conv2D(filters[f],
                                      kernel_size=(3, 3),
                                      padding='same',
                                      activation='relu')(decoded)
        decoded = keras.layers.UpSampling2D((2, 2))(decoded)
    x = keras.layers.Conv2D(filters[0],
                            kernel_size=(3, 3),
                            padding='valid',
                            activation='relu')(decoded)
    x = keras.layers.UpSampling2D((2, 2))(x)
    x = keras.layers.Conv2D(input_dims[-1],
                            kernel_size=(3, 3),
                            padding='same',
                            activation='sigmoid')(x)
    decoder = keras.Model(inputs=input_decoder, outputs=x)

    autoencoder = keras.Model(
        input_img, decoder(encoder(input_img)), name='autoencoder'
    )
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    return encoder, decoder, autoencoder
