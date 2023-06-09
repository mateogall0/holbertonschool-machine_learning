#!/usr/bin/env python3
"""
    Train
"""


import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False, patience=0,
                verbose=True, shuffle=False):
    """
        Trains a model using mini-batch gradient descent

        network -- model to train
        data -- numpy.ndarray of shape (m, nx) containing the input data
        labels -- one-hot numpy.ndarray of shape (m, classes) containing
        the labels of data
        batch_size -- size of the batch used for mini-batch gradient descent
        epochs -- number of passes through data for mini-batch gradient
        descent
        verbose -- boolean that determines if output should be printed during
        training
        shuffle -- boolean that determines whether to shuffle the batches
        every epoch. Normally, it is a good idea to shuffle, but for
        reproducibility, we have chosen to set the default to False.
        early_stopping -- boolean that indicates whether early stopping should
        be used
            early stopping should only be performed if validation_data exists
            early stopping should be based on validation loss

        Returns: the History object generated after training the model
    """
    if validation_data is not None and early_stopping:
        callbacks = [K.callbacks.EarlyStopping(monitor='val_loss',
                                               patience=patience)]
    else:
        callbacks = None
    return network.fit(x=data,
                       y=labels,
                       batch_size=batch_size,
                       epochs=epochs,
                       verbose=verbose,
                       shuffle=shuffle,
                       validation_data=validation_data,
                       callbacks=callbacks)
