#!/usr/bin/env python3
"""
K-means
"""


import sklearn.cluster


def kmeans(X, k):
    """
    K-means
    """
    kmeans = sklearn.cluster.KMeans(n_clusters=k)
    kmeans.fit(X)
    return kmeans.cluster_centers_, kmeans.labels_
