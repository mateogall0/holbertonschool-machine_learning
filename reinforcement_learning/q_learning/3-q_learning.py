#!/usr/bin/env python3

import numpy as np
epsilon_greedy = __import__('2-epsilon_greedy').epsilon_greedy


def train(env, Q, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99,
          lon=1, min_epsilon=0.1, epsilon_decay=0.05):
    total_rewards_list = []

    for episode in range(episodes):
        state = env.reset()
        total_rewards = 0
        epsilon = min_epsilon + (1 - min_epsilon) * np.exp(-epsilon_decay * episode)

        for _ in range(max_steps):
            action = epsilon_greedy(Q, state, epsilon)
            new_state, reward, done, _ = env.step(action)

            Q[state, action] += alpha * (reward + gamma * np.max(Q[new_state, :]) - Q[state, action])
            state = new_state
            total_rewards += reward

            if done:
                break

        total_rewards_list.append(total_rewards)

    return Q, total_rewards_list
