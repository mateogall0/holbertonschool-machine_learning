#!/usr/bin/env python3

import numpy as np

def monte_carlo(env, V, policy, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99):
    for episode in range(episodes):
        state = env.reset()[0]
        episode_states = []
        episode_rewards = []

        for _ in range(max_steps):
            action = policy(state)
            next_state, reward, done, _, _ = env.step(action)
            episode_states.append(state)
            episode_rewards.append(reward)

            if done:
                break
            state = next_state

        G = 0
        for t in range(len(episode_states) - 1, -1, -1):
            G = gamma * G + episode_rewards[t]
            V[episode_states[t]] = V[episode_states[t]] + alpha * (G - V[episode_states[t]])

    return V
