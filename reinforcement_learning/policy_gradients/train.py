#!/usr/bin/env python3

from policy_gradient import policy_gradient
import numpy as np

def train(env, nb_episodes, alpha=0.000045, gamma=0.98, initial_weight=None, show_result=False):
    scores = []
    
    if initial_weight is None:
        weight = np.random.rand(env.action_space.n)
    else:
        weight = initial_weight

    for episode in range(1, nb_episodes + 1):
        state = env.reset()
        episode_score = 0
        episode_states = []
        episode_gradients = []

        while True:
            action, gradient = policy_gradient(state, weight)
            next_state, reward, done, _ = env.step(action)
            episode_score += reward
            episode_states.append(state)
            episode_gradients.append(gradient)

            if done:
                break

            state = next_state

        for t, gradient in enumerate(episode_gradients):
            G = np.sum([gamma**i * episode_score for i in range(t, len(episode_states))])
            weight += alpha * G * gradient

        scores.append(episode_score)
        print(f"Episode {episode}/{nb_episodes}, Score: {episode_score}", end="\r", flush=True)

        if show_result and episode % 1000 == 0:
            env.render()

    return scores