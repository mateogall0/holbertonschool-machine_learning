#!/usr/bin/env python3


def monte_carlo(env, V, policy, episodes=5000, max_steps=100, alpha=0.1,
                gamma=0.99):
    """
    Performs the Monte Carlo algorithm
    """
    for _ in range(episodes):
        state = env.reset()[0]
        episode = []
        for _ in range(max_steps):
            action = policy(state)
            next_state, reward, done, _, _ = env.step(action)
            episode.append((state, action, reward))
            if done:
                break
            state = next_state

        G = 0
        states_and_returns = []
        for state, action, reward in reversed(episode):
            G = reward + gamma * G
            states_and_returns.append((state, G))
        states_and_returns.reverse()

        visited_states = set()
        for state, G in states_and_returns:
            if state not in visited_states:
                visited_states.add(state)
                V[state] += alpha * (G - V[state])
    return V
