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
    X = K.Input(shape=(224, 224, 3))

    conv1 = K.layers.Conv2D(64,
                            kernel_size=(7, 7),
                            strides=(2, 2),
                            padding="same", kernel_initializer='he_normal')(X)
    norm1 = K.layers.BatchNormalization()(conv1)
    act1 = K.layers.Activation("relu")(norm1)

    pool0 = K.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2),
                                     padding="same")(act1)

    # Conv2_x blocks
    conv2_x_1 = projection_block(pool0, [64, 64, 256], 1)

    conv2_x_2 = identity_block(conv2_x_1, [64, 64, 256])
    conv2_x_3 = identity_block(conv2_x_2, [64, 64, 256])

    # Conv3_x blocks
    conv3_x_1 = projection_block(conv2_x_3, [128, 128, 512])

    conv3_x_2 = identity_block(conv3_x_1, [128, 128, 512])
    conv3_x_3 = identity_block(conv3_x_2, [128, 128, 512])
    conv3_x_4 = identity_block(conv3_x_3, [128, 128, 512])

    # Conv4_x blocks

    conv4_x_1 = projection_block(conv3_x_4, [256, 256, 1024])

    conv4_x_2 = identity_block(conv4_x_1, [256, 256, 1024])
    conv4_x_3 = identity_block(conv4_x_2, [256, 256, 1024])
    conv4_x_4 = identity_block(conv4_x_3, [256, 256, 1024])
    conv4_x_5 = identity_block(conv4_x_4, [256, 256, 1024])
    conv4_x_6 = identity_block(conv4_x_5, [256, 256, 1024])

    # Conv5_x blocks
    conv5_x_1 = projection_block(conv4_x_6, [512, 512, 2048])

    conv5_x_2 = identity_block(conv5_x_1, [512, 512, 2048])
    conv5_x_3 = identity_block(conv5_x_2, [512, 512, 2048])

    pool1 = K.layers.AveragePooling2D(pool_size=(7, 7),
                                         strides=(1, 1),
                                         padding="valid")(conv5_x_3)

    softmax = K.layers.Dense(1000,
                             activation="softmax",
                             kernel_initializer='he_normal')(pool1)

    model = K.Model(inputs=X, outputs=softmax)

    return model
