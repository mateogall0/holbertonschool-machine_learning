#!/usr/bin/env python3
"""
Inception Network
"""


inception_block = __import__('0-inception_block').inception_block
import tensorflow.keras as K


def inception_network():
    """
    Builds the inception network as described in Going Deeper with
    Convolutions (2014)

    You can assume the input data will have shape (224, 224, 3)
    All convolutions inside and outside the inception block should use a
    rectified linear activation (ReLU)
    Returns: the keras model
    """
    inputs = K.layers.Input(shape=(224, 224, 3))

    # Stage 1
    conv1 = K.layers.Conv2D(64, (7, 7), strides=(2, 2), padding='same', activation='relu')(inputs)
    pool1 = K.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(conv1)

    # Stage 2
    conv2_1 = K.layers.Conv2D(64, (1, 1), padding='same', activation='relu')(pool1)
    conv2_2 = K.layers.Conv2D(192, (3, 3), padding='same', activation='relu')(conv2_1)
    pool2 = K.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(conv2_2)

    # Stage 3
    inception3a = inception_block(pool2, [64, 96, 128, 16, 32, 32])
    inception3b = inception_block(inception3a, [128, 128, 192, 32, 96, 64])
    pool3 = K.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(inception3b)

    # Stage 4
    inception4a = inception_block(pool3, [192, 96, 208, 16, 48, 64])
    inception4b = inception_block(inception4a, [160, 112, 224, 24, 64, 64])
    inception4c = inception_block(inception4b, [128, 128, 256, 24, 64, 64])
    inception4d = inception_block(inception4c, [112, 144, 288, 32, 64, 64])
    inception4e = inception_block(inception4d, [256, 160, 320, 32, 128, 128])
    pool4 = K.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(inception4e)

    # Stage 5
    inception5a = inception_block(pool4, [256, 160, 320, 32, 128, 128])
    inception5b = inception_block(inception5a, [384, 192, 384, 48, 128, 128])

    # Average pooling
    avg_pool = K.layers.GlobalAveragePooling2D()(inception5b)

    # Fully connected layer
    fc = K.layers.Dense(1000, activation='softmax')(avg_pool)

    return K.models.Model(inputs=inputs, outputs=fc)
