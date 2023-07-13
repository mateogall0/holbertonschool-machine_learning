#!/usr/bin/env python3
"""
K-means
"""


import numpy as np


def kmeans(X, k, iterations=1000):
    """
    Performs K-means on a dataset

    X -- numpy.ndarray of shape (n, d) containing the dataset
        n -- number of data points
        d -- number of dimensions for each data point
    k -- positive integer containing the number of clusters
    iterations -- positive integer containing the maximum number of iterations
    that should be performed
    If no change in the cluster centroids occurs between iterations, your
    function should return
    Initialize the cluster centroids using a multivariate uniform distribution
    (based on0-initialize.py)
    If a cluster contains no data points during the update step, reinitialize
    its centroid

    Returns: C, clss, or None, None on failure
        C -- numpy.ndarray of shape (k, d) containing the centroid means for
        each cluster
        clss -- numpy.ndarray of shape (n,) containing the index of the
        cluster in C that each data point belongs to
    """
    try:
        centroids = initialize(X, k)
        for _ in range(iterations):
            centroids_copy = centroids.copy()
            points_centroids_distance = np.sqrt(np.sum((X - centroids[:, np.newaxis]) ** 2, axis=2))
            clss = np.argmin(points_centroids_distance, axis=0)

            for j in range(k):
                if len(X[clss == j]) == 0:
                    centroids[j] = initialize(X, 1)
                else:
                    centroids[j] = np.mean(X[clss == j], axis=0)

            points_centroids_distance = np.sqrt(np.sum((X - centroids[:, np.newaxis]) ** 2, axis=2))
            clss = np.argmin(points_centroids_distance, axis=0)
            if np.all(centroids_copy == centroids):
                break

        return centroids, clss
    except Exception:
        return None, None

def initialize(X, k):
    """
    Initializes cluster centroids for K-means

    X -- numpy.ndarray of shape (n, d) containing the dataset that will be
    used for K-means clustering
        n -- number of data points
        d -- number of dimensions for each data point
    k -- positive integer containing the number of clusters
    The cluster centroids should be initialized with a multivariate uniform
    distribution along each dimension in d:
    The minimum values for the distribution should be the minimum values of X
    along each dimension in d
    The maximum values for the distribution should be the maximum values of X
    along each dimension in d

    Returns: a numpy.ndarray of shape (k, d) containing the initialized
    centroids for each cluster, or None on failure
    """
    if not isinstance(k, int) or k <= 0:
        return
    try:
        min = np.min(X, axis=0)
        max = np.max(X, axis=0)

        # Obtain the centroids
        return np.random.uniform(min, max, size=(k, X.shape[1]))
    except Exception:
        pass
