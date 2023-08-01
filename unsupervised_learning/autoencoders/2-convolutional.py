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
        x = keras.layers.Conv2D(f, (3, 3), activation='relu', padding='same')(x)
        x = keras.layers.MaxPooling2D((2, 2), padding='same')(x)
    encoder = keras.Model(input_img, x, name='encoder')
    
    # Decoder
    latent_input = keras.Input(shape=latent_dims)
    x = latent_input
    for f in reversed(filters[:-1]):
        x = keras.layers.Conv2D(f, (3, 3), activation='relu', padding='same')(x)
        x = keras.layers.UpSampling2D((2, 2))(x)
    x = keras.layers.Conv2D(filters[-1], (3, 3), activation='relu', padding='valid')(x)
    x = keras.layers.UpSampling2D((2, 2))(x)
    decoded = keras.layers.Conv2D(input_dims[-1], (3, 3), activation='sigmoid', padding='same')(x)
    decoder = keras.Model(latent_input, decoded, name='decoder')
    
    # Autoencoder
    autoencoder = keras.Model(input_img, decoder(encoder(input_img)), name='autoencoder')
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    
    return encoder, decoder, autoencoder