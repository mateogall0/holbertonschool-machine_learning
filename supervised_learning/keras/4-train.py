#!/usr/bin/env python3
"""
    Train
"""


import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs, verbose=True,
                shuffle=False):
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

        Returns: the History object generated after training the model
    """
    history = K.callbacks.History()

    network.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

    for epoch in range(epochs):
        if shuffle:
            permutation = np.random.permutation(data.shape[0])
            data = data[permutation]
            labels = labels[permutation]
            
        for i in range(0, data.shape[0], batch_size):
            batch_data = data[i:i+batch_size]
            batch_labels = labels[i:i+batch_size]
            network.train_on_batch(batch_data, batch_labels)

        if verbose:
            loss, accuracy = network.evaluate(data, labels, verbose=0)
            print('Epoch {}/{} - loss: {:.4f} - acc: {:.4f}'.format(epoch+1, epochs, loss, accuracy))

    return history
