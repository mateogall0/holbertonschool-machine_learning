#!/usr/bin/env python3
"""
Variational
"""


import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Creates a variational autoencoder
    """
    encoder_inputs = keras.Input(shape=(input_dims,))
    x = encoder_inputs
    for nodes in hidden_layers:
        x = keras.layers.Dense(nodes, activation='relu')(x)
    z_mean = keras.layers.Dense(latent_dims)(x)
    z_log_var = keras.layers.Dense(latent_dims)(x)

    def sampling(args):
        z_mean, z_log_var = args
        epsilon = keras.backend.random_normal(
            shape=(keras.backend.shape(z_mean)[0], latent_dims),
            mean=0.0, stddev=1.0
        )
        return z_mean + keras.backend.exp(0.5 * z_log_var) * epsilon

    z = keras.layers.Lambda(sampling)([z_mean, z_log_var])
    encoder = keras.Model(
        encoder_inputs, [z, z_mean, z_log_var], name="encoder"
    )
    latent_inputs = keras.Input(shape=(latent_dims,))
    x = latent_inputs
    for nodes in reversed(hidden_layers):
        x = keras.layers.Dense(nodes, activation='relu')(x)
    decoder_outputs = keras.layers.Dense(input_dims, activation='sigmoid')(x)
    decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")

    autoencoder_outputs = decoder(encoder(encoder_inputs)[0])
    autoencoder = keras.Model(
        encoder_inputs, autoencoder_outputs, name="autoencoder"
    )

    def custom_loss(x, decoded_x):
        reconstruction_loss = keras.backend.sum(
            keras.backend.binary_crossentropy(x, decoded_x), axis=1
        )
        kl_loss = -0.5 * keras.backend.mean(
            1 + z_log_var - keras.backend.square(z_mean) - keras.backend.exp(
                z_log_var), axis=-1
        )
        return reconstruction_loss + kl_loss

    autoencoder.compile(optimizer='adam', loss=custom_loss)
    return encoder, decoder, autoencoder
