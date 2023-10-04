#!/usr/bin/env python3

import numpy as np


def play(env, Q, max_steps=100):
    state = env.reset()
    env.render()
    for _ in range(max_steps):
        action = np.argmax(Q[state])
        new_state, reward, done, _ = env.step(action)
        env.render()
        if done:
            return reward
        state = new_state
    env.close()    
