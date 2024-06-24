#!/usr/bin/env python3
epsilon_greedy = __import__('2-epsilon_greedy').epsilon_greedy
import numpy as np


def play(env, Q, max_steps=100):
    """
    Makes the agent play an episode
    """
    state = env.reset()[0]
    done = False
    step = 0
    rewards = 0
    while not done and step < max_steps:
        env.render()
        action = np.argmax(Q[state, :])
        new_state, reward, done, _, _ = env.step(action)
        state = new_state
        step += 1
        rewards += reward
    return rewards
