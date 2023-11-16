#!/usr/bin/env python3
"""
Task 34
"""


from pymongo import MongoClient


if __name__ == "__main__":
    client = MongoClient('mongodb://127.0.0.1:27017')
    logs = client.logs.nginx.count_documents({})
    print("{} logs".format(logs))
    print("Methods:")
    methods = ["GET", "POST", "PUT", "PATCH", "DELETE"]
    for m in methods:
        count = logs.count_documents({"method": m})
        print("   method {}: {}".format(method, count))
    filter_path = {"method": "GET", "path": "/status"}
    path_count = logs.count_documents(filter_path)
    print("{} status check".format(path_count))
