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
    initializer = K.initializers.he_normal()
    input_data = K.Input(shape=(224, 224, 3))

    conv0 = K.layers.Conv2D(filters=64,
                            kernel_size=(7, 7),
                            strides=(2, 2),
                            padding="same",
                            kernel_initializer=initializer)(input_data)
    batch1 = K.layers.BatchNormalization(axis=3)(conv0)
    activation0 = K.layers.ReLU()(batch1)
    pool0 = K.layers.MaxPooling2D(pool_size=(3, 3),
                                  strides=(2, 2),
                                  padding="same")(activation0)

    prblock_1_2x = projection_block(pool0, [64, 64, 256], s=1)
    idblock_1_2x = identity_block(prblock_1_2x, [64, 64, 256])
    idblock_2_2x = identity_block(idblock_1_2x, [64, 64, 256])
    prblock_1_3x = projection_block(idblock_2_2x, [128, 128, 512])
    idblock_1_3x = identity_block(prblock_1_3x, [128, 128, 512])
    idblock_2_3x = identity_block(idblock_1_3x, [128, 128, 512])
    idblock_3_3x = identity_block(idblock_2_3x, [128, 128, 512])
    prblock_1_4x = projection_block(idblock_3_3x, [256, 256, 1024])
    idblock_1_4x = identity_block(prblock_1_4x, [256, 256, 1024])
    idblock_2_4x = identity_block(idblock_1_4x, [256, 256, 1024])
    idblock_3_4x = identity_block(idblock_2_4x, [256, 256, 1024])
    idblock_4_4x = identity_block(idblock_3_4x, [256, 256, 1024])
    idblock_5_4x = identity_block(idblock_4_4x, [256, 256, 1024])
    prblock_1_5x = projection_block(idblock_5_4x, [512, 512, 2048])
    idblock_1_5x = identity_block(prblock_1_5x, [512, 512, 2048])
    idblock_2_5x = identity_block(idblock_1_5x, [512, 512, 2048])

    pool1 = K.layers.AvgPool2D(pool_size=(7, 7),
                               strides=(1, 1),
                               padding="valid")(idblock_2_5x)
    softmax = K.layers.Dense(units=1000,
                             activation="softmax",
                             kernel_initializer=initializer)(pool1)
    return K.models.Model(inputs=input_data, outputs=softmax)
