#!/usr/bin/env python3

import numpy as np


def sarsa_lambtha(env, Q, lambtha, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99, epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    for _ in range(episodes):
        state = env.reset()[0]
        eligibility_trace = np.zeros_like(Q)
        action = epsilon_greedy_policy(Q, state, epsilon)

        for _ in range(max_steps):
            next_state, reward, done, _, _ = env.step(action)
            next_action = epsilon_greedy_policy(Q, next_state, epsilon)

            delta = reward + gamma * Q[next_state, next_action] - Q[state, action]
            eligibility_trace[state, action] += 1
            Q += alpha * delta * eligibility_trace

            eligibility_trace *= lambtha * gamma

            if done:
                break

            state = next_state
            action = next_action

        epsilon = max(epsilon - epsilon_decay, min_epsilon)

    return Q

def epsilon_greedy_policy(Q, state, epsilon):
    if np.random.rand() < epsilon:
        return np.random.choice(Q.shape[1])
    return np.argmax(Q[state])
