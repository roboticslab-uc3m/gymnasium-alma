#!/usr/bin/env python

import gymnasium as gym
import gymnasium_playground_tiago_ironing

env = gym.make('gymnasium_playground/TiagoIroning-v0',
               render_mode='human',  # "human", "text", None
               inFileStr='../assets/map1.csv',
               initX=2,
               initY=2)
env.reset()
env.render()
