#!/usr/bin/env python3
import numpy as np
epsilon_greedy = __import__('2-epsilon_greedy').epsilon_greedy


def train(env, Q, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99,
          epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """
    Performs Q-learning
    """
    total_rewards = []
    for _ in range(episodes):
        state = env.reset()[0]
        done = False
        total_rewards.append(-1)
        step = 0
        while not done and step < max_steps:
            action = epsilon_greedy(Q, state, epsilon)
            new_state, reward, done, truncated, info = env.step(action)
            Q[state, action] = (
                Q[state, action] + alpha *
                (reward + gamma * np.max(Q[new_state]) - Q[state, action])
            )
            state = new_state
            step += 1
            if reward > 0:
                total_rewards[-1] = reward
            epsilon = max(min_epsilon, epsilon * epsilon_decay)
    return Q, total_rewards
