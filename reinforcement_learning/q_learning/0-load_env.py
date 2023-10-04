#!/usr/bin/env python3

import gym


def load_frozen_lake(desc=None, map_name=None, is_slippery=False):
    """if map_name is not None:
        env = gym.make('FrozenLake-v1', is_slippery=is_slippery)
    elif desc is not None:
        env = gym.make('FrozenLake-v1', desc=desc, is_slippery=is_slippery)
    else:
        env = gym.make('FrozenLake-v1', is_slippery=is_slippery)
    return env"""
    return gym.make('FrozenLake-v1', map_name=map_name, desc=desc,
                    is_slippery=is_slippery)
