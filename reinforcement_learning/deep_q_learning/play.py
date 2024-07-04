#!/usr/bin/env python3
import gym
from keras.layers import Dense, Flatten, Conv2D, Permute, Input
from keras.models import Model
from keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import EpsGreedyQPolicy

WINDOW_LENGTH = 4
env = gym.make('Breakout-v5')
INPUT_SHAPE = env.observation_space.shape
nb_actions = env.action_space.n

input_shape = INPUT_SHAPE
X = Input(input_shape)
X_permuted = Permute((2, 1, 3))(X)
X_conv1 = Conv2D(32, (8, 8), strides=(4, 4), activation='relu')(X_permuted)
X_conv2 = Conv2D(64, (4, 4), strides=(2, 2), activation='relu')(X_conv1)
X_conv3 = Conv2D(64, (3, 3), strides=(1, 1), activation='relu')(X_conv2)
X_flattened = Flatten()(X_conv3)
X_dense1 = Dense(512, activation='relu')(X_flattened)
X_output = Dense(nb_actions, activation='linear')(X_dense1)

model = Model(inputs=X, outputs=X_output)

memory = SequentialMemory(limit=1000000, window_length=WINDOW_LENGTH)
policy = EpsGreedyQPolicy()
dqn = DQNAgent(model=model,
               policy=policy,
               nb_actions=nb_actions,
               memory=memory,
               target_model_update=10000,
               nb_steps_warmup=1000)

dqn.load_weights('policy.h5')
dqn.test(env, nb_episodes=10, visualize=True)