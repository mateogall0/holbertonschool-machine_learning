#!/usr/bin/env python3
"""
Rate me is you can!
"""

import time
import sys
import requests

if __name__ == '__main__':
    if len(sys.argv) != 2:
        exit()
    url = sys.argv[1]
    headers = {'Accept': 'application/vnd.github.v3+json'}
    res = requests.get(url, headers=headers)
    if res.status_code == 200:
        print(res.json()['location'])
    elif res.status_code == 403:
        rate_limit = int(res.headers['X-Ratelimit-Reset'])
        now = int(time.time())
        minutes = int((rate_limit - now) / 60)
        print("Reset in {} min".format(minutes))
    elif res.status_code == 404:
        print('Not found')
