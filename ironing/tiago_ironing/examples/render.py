#!/usr/bin/env python

import gymnasium as gym
import gymnasium_alma_tiago_ironing

env = gym.make('gymnasium_alma/TiagoIroning-v0',
               render_mode='human',  # "human", "text", None
               inFileStr='../assets/map1.csv',
               initX=2,
               initY=2)
env.reset()
env.render()
