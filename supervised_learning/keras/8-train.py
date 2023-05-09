#!/usr/bin/env python3
"""
    Train
"""


import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False, patience=0,
                learning_rate_decay=False, alpha=0.1, decay_rate=1,
                save_best=False, filepath=None, verbose=True, shuffle=False):
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
        learning_rate_decay -- boolean that indicates whether learning rate
        decay should be used
            learning rate decay should only be performed if validation_data
            exists
            the decay should be performed using inverse time decay
            the learning rate should decay in a stepwise fashion after each
            epoch
            each time the learning rate updates, Keras should print a message
        alpha -- initial learning rate
        decay_rate -- decay rate

        save_best -- boolean indicating whether to save the model after each
        epoch if it is the best
            a model is considered the best if its validation loss is the
            lowest that the model has obtained
        filepath -- file path where the model should be saved

        Returns: the History object generated after training the model
    """
    callbacks = []
    if validation_data is not None and early_stopping:
        callbacks.append(K.callbacks.EarlyStopping(monitor='val_loss',
                                                   patience=patience))

    if validation_data is not None and learning_rate_decay:
        def lr_schedule(epoch):
            return alpha / (1 + decay_rate * epoch)

        callbacks.append(K.callbacks.LearningRateScheduler(lr_schedule,
                                                           verbose=True))

    if validation_data is not None and save_best and filepath:
        callbacks.append(
            K.callbacks.ModelCheckpoint(filepath, save_best_only=True)
            )

    if len(callbacks) == 0:
        callbacks = None

    return network.fit(x=data,
                       y=labels,
                       batch_size=batch_size,
                       epochs=epochs,
                       verbose=verbose,
                       shuffle=shuffle,
                       validation_data=validation_data,
                       callbacks=callbacks)
