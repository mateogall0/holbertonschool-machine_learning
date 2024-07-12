#!/usr/bin/env python3
import numpy as np


def td_lambtha(env, V, policy, lambtha, episodes=5000, max_steps=100,
               alpha=0.1, gamma=0.99):
    """
    Performs the TD(λ) algorithm
    """
    for _ in range(episodes):
        state = env.reset()[0]
        eligibility_traces = np.zeros_like(V)

        for _ in range(max_steps):
            action = policy(state)
            next_state, reward, done, _, _ = env.step(action)

            delta = reward + gamma * V[next_state] - V[state]
            eligibility_traces[state] += 1

            V += alpha * delta * eligibility_traces
            eligibility_traces *= gamma * lambtha

            if done:
                break

            state = next_state

    return V
