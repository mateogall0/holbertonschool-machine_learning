#!/usr/bin/env python3
"""
Task 33
"""


def schools_by_topic(mongo_collection, topic):
    """
    Write a Python function that returns the list of school having a specific
    topic
    """
    return mongo_collection.find({"topics": {"$in": [topic]}})
