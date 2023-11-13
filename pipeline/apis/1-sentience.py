#!/usr/bin/env python3
"""
Where I am?
"""
import requests


def sentientPlanets():
    """
    By using the Swapi API, create a method that returns the list of names of
    the home planets of all sentient species.
    """
    url = "https://swapi-api.hbtn.io/api/species/?format=json"
    all_species = []
    while url:
        results = requests.get(url).json()
        all_species += results.get('results')
        url = results.get('next')
    all_planets = []
    for species in all_species:
        if species.get('designation') == 'sentient' or \
           species.get('classification') == 'sentient':
            url = species.get('homeworld')
            if url:
                planet = requests.get(url).json()
                all_planets.append(planet.get('name'))
    return all_planets
