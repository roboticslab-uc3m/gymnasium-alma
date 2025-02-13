#!/usr/bin/env python

import gymnasium as gym
import gymnasium_alma_gridworld

import numpy as np
import time

"""
# Coordinate Systems for `.csv` and `print(numpy)`

X points down (rows); Y points right (columns); Z would point outwards.

*--> Y (columns)
|
v
X (rows)
"""
UP = 0
UP_RIGHT = 1
RIGHT = 2
DOWN_RIGHT = 3
DOWN = 4
DOWN_LEFT = 5
LEFT = 6
UP_LEFT = 7

SIM_PERIOD_MS = 500.0

env = gym.make('gymnasium_alma/GridWorld-v0',
               render_mode='human',  # "human", "text", None
               inFileStr='../assets/map1.csv',
               initX=2,
               initY=2,
               goalX=7,
               goalY=2)
observation, info = env.reset()
print("observation: "+str(observation)+", info: "+str(info))
env.render()
time.sleep(0.5)

for i in range(5):
    observation, reward, terminated, truncated, info = env.step(DOWN_RIGHT)
    env.render()
    print("observation: " + str(observation)+", reward: " + str(reward) + ", terminated: " +
          str(terminated) + ", truncated: " + str(truncated) + ", info: " + str(info))
    time.sleep(SIM_PERIOD_MS/1000.0)
