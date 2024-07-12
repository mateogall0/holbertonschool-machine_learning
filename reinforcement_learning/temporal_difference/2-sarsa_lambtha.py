#!/usr/bin/env python3
import numpy as np


def sarsa_lambtha(env, Q, lambtha, episodes=5000, max_steps=100, alpha=0.1,
                  gamma=0.99, epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """
    Performs SARSA(λ)
    """
    def epsilon_greedy(state, epsilon):
        if np.random.rand() < epsilon:
            return np.random.choice(env.action_space.n)
        else:
            return np.argmax(Q[state])

    for _ in range(episodes):
        state = env.reset()[0]
        action = epsilon_greedy(state, epsilon)
        eligibility_traces = np.zeros_like(Q)

        for _ in range(max_steps):
            next_state, reward, done, _, _ = env.step(action)
            next_action = epsilon_greedy(next_state, epsilon)

            delta = (reward + gamma *
                     Q[next_state, next_action] -
                     Q[state, action])
            eligibility_traces[state, action] += 1

            Q += alpha * delta * eligibility_traces
            eligibility_traces *= gamma * lambtha

            if done:
                break

            state = next_state
            action = next_action

        epsilon = max(min_epsilon, epsilon * np.exp(-epsilon_decay))

    return Q
