#!/usr/bin/env python3

from gensim.models import Word2Vec

def gensim_to_keras(model):
    return model.wv.get_keras_embedding(train_embeddings=False)