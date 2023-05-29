#!/usr/bin/env python3
"""
Transfer Knowledge
"""


import numpy as np
import tensorflow.keras as K
import tensorflow as tf

def preprocess_data(X, Y):
    X_p = K.applications.resnet50.preprocess_input(X)
    Y_p = K.utils.to_categorical(Y, 10)
    return X_p, Y_p


if __name__ == '__main__':
    model = K.models.Sequential()
    input_t = K.Input(shape=(32, 32, 3))
    resnet = K.applications.ResNet50(include_top=False,
                                     weights="imagenet",
                                     input_tensor=input_t)
    for layer in resnet.layers[:143]:
        layer.trainable = False
    model.add(K.layers.Lambda(lambda x: K.backend.resize_images(x, 2, 2, 'channels_last'),
                 input_shape=(32, 32, 3)))
    model.add(resnet)
    model.add(K.layers.Flatten())
    model.add(K.layers.BatchNormalization())
    model.add(K.layers.Dense(256, activation='relu'))
    model.add(K.layers.Dropout(0.5))
    model.add(K.layers.BatchNormalization())
    model.add(K.layers.Dense(128, activation='relu'))
    model.add(K.layers.Dropout(0.5))
    model.add(K.layers.BatchNormalization())
    model.add(K.layers.Dense(64, activation='relu'))
    model.add(K.layers.Dropout(0.5))
    model.add(K.layers.BatchNormalization())
    model.add(K.layers.Dense(10, activation='softmax'))

    model.compile(optimizer=K.optimizers.RMSprop(lr=2e-5),
                  loss='categorical_crossentropy', metrics=['accuracy'])

    (X, Y), (X_test, Y_test) = K.datasets.cifar10.load_data()
    X_p, Y_p = preprocess_data(X, Y)
    X_test, Y_test = preprocess_data(X_test, Y_test)

    callbacks = []

    callbacks.append(K.callbacks.ModelCheckpoint(
        filepath="cifar10.h5",
        monitor="val_acc",
        mode="max",
        save_best_only=True,
        ))

    model.fit(X_p, Y_p,
              batch_size=512, epochs=15,
              callbacks=callbacks,
              validation_data=(X_test, Y_test))

    model.save('cifar10.h5')
