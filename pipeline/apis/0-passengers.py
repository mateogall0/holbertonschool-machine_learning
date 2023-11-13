#!/usr/bin/env python3
"""
Can I join?
"""
import requests


def availableShips(passengerCount):
    """
    By using the Swapi API, create a method that returns the list of ships
    that can hold a given number of passengers
    """
    url = "https://swapi-api.hbtn.io/api/starships/"
    results = requests.get(url).json()
    all_starships = []
    while results["next"]:
        res = results['results']
        for r in res:
            if r["passengers"] == 'n/a' or r["passengers"] == 'unknown':
                continue
            if int(r["passengers"].replace(',', '')) >= passengerCount:
                all_starships.append(r["name"])
        url = results['next']
        results = requests.get(url).json()
    return all_starships
