#!/usr/bin/env python3
"""
By using the (unofficial) SpaceX API, write a script that displays the first
launch with these information:

Name of the launch
The date (in local time)
The rocket name
The name (with the locality) of the launchpad
"""
import requests


def get_json(url):
    """Request JSON"""
    response = requests.get(url)
    return response.json()


def get_upcoming_launch():
    """Upcoming launch"""
    url = "https://api.spacexdata.com/v4/launches/upcoming"
    launches = get_json(url)

    dates = [launch['date_unix'] for launch in launches]
    min_date_index = dates.index(min(dates))
    next_launch = launches[min_date_index]

    return next_launch


def get_rocket_name(rocket_id):
    """Rocket name"""
    url = "https://api.spacexdata.com/v4/rockets/{}".format(rocket_id)
    rocket_data = get_json(url)
    return rocket_data['name']


def get_launchpad_info(launchpad_id):
    """Launchpad info"""
    url = "https://api.spacexdata.com/v4/launchpads/{}".format(launchpad_id)
    launchpad_data = get_json(url)
    return launchpad_data['name'], launchpad_data['locality']


if __name__ == '__main__':
    next_launch = get_upcoming_launch()

    name = next_launch['name']
    date = next_launch['date_local']
    rocket_name = get_rocket_name(next_launch['rocket'])
    launchpad_name, launchpad_loc = get_launchpad_info(
        next_launch['launchpad']
    )

    print("{} ({}) {} - {} ({})".format(name,
                                        date,
                                        rocket_name,
                                        launchpad_name,
                                        launchpad_loc))
