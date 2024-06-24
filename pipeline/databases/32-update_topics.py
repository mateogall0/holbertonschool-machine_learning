#!/usr/bin/env python3
"""
Task 32
"""


def update_topics(mongo_collection, name, topics):
    """
    Write a Python function that changes all topics of a school document based
    on the name
    """
    search = {"name": name}
    new = {"$set": {"topics": topics}}

    mongo_collection.update_many(search, new)
