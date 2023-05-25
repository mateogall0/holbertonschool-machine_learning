#!/usr/bin/env python3
"""
ResNet-50
"""


import tensorflow.keras as K
identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """
    Builds the ResNet-50 architecture as described in Deep Residual Learning
    for Image Recognition (2015):

    You can assume the input data will have shape (224, 224, 3)
    All convolutions inside and outside the blocks should be followed by batch
    normalization along the channels axis and a rectified linear activation
    (ReLU), respectively.
    All weights should use he normal initialization
    Returns: the keras model
    """
    input_shape = (224, 224, 3)

    X_input = K.Input(input_shape)
    initializer = K.initializers.he_normal()

    # Stage 1
    X = K.layers.Conv2D(64, (1, 1), strides=(2, 2),
                        kernel_initializer=initializer)(X_input)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)
    X = K.layers.MaxPooling2D((3, 3), strides=(2, 2))(X)

    # Stage 2
    X = projection_block(X, [64, 64, 256], 1)
    X = identity_block(X, [64, 64, 256])
    X = identity_block(X, [64, 64, 256])

    # Stage 3
    X = projection_block(X, [128, 128, 512], 2)
    X = identity_block(X, [128, 128, 512])
    X = identity_block(X, [128, 128, 512])
    X = identity_block(X, [128, 128, 512])

    # Stage 4
    X = projection_block(X, [256, 256, 1024], 3)
    X = identity_block(X, [256, 256, 1024])
    X = identity_block(X, [256, 256, 1024])
    X = identity_block(X, [256, 256, 1024])
    X = identity_block(X, [256, 256, 1024])
    X = identity_block(X, [256, 256, 1024])

    # Stage 5
    X = projection_block(X, [512, 512, 2048], 4)
    X = identity_block(X, [512, 512, 2048])
    X = identity_block(X, [512, 512, 2048])

    # Average pooling
    X = K.layers.AveragePooling2D()(X)

    # Output layer
    X = K.layers.Dense(1000, activation='softmax')(X)

    # Create model
    return K.models.Model(inputs=X_input, outputs=X)
