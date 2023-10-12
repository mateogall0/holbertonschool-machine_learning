#!/usr/bin/env python3

import gym
from keras.models import Sequential
from rl.agents.dqn import DQNAgent
from rl.policy import GreedyQPolicy

env = gym.make('Breakout-v0')

input_shape = (1,) + env.observation_space.shape

nb_actions = env.action_space.n

model = Sequential()
model.add(Flatten(input_shape=input_shape))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(nb_actions, activation='linear'))

policy = GreedyQPolicy()
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=None, nb_steps_warmup=1000, target_model_update=1e-2, policy=policy)

dqn.load_weights('policy.h5')

dqn.test(env, nb_episodes=5, visualize=True)
