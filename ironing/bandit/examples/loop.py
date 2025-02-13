#!/usr/bin/env python

import gymnasium as gym
import gymnasium_alma_bandit

import numpy as np
import time

env = gym.make('gymnasium_alma/Bandit-v0',
               render_mode='human', n_arms=10)
observation, info = env.reset()
print("observation: "+str(observation)+", info: "+str(info))
env.render()

for i in range(5):
    observation, reward, terminated, truncated, info = env.step(i)
    env.render()
    print("observation: " + str(observation)+", reward: " + str(reward) + ", terminated: " +
          str(terminated) + ", truncated: " + str(truncated) + ", info: " + str(info))
