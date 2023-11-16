#!/usr/bin/env python3
"""
Task 30
"""


def list_all(mongo_collection):
    """
    Write a Python function that lists all documents in a collection
    """
    return list(mongo_collection.find())
