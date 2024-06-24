#!/usr/bin/env python3
"""
Task 31
"""


def insert_school(mongo_collection, **kwargs):
    """
    Write a Python function that inserts a new document in a collection based
    on kwargs
    """
    return mongo_collection.insert_one(kwargs).inserted_id
