#!/usr/bin/env python

import gymnasium as gym
import gymnasium_playground_folding

env = gym.make('gymnasium_playground/Folding-v0',
               render_mode='human',  # "human", "text", None
               inFileLabelsStr='../assets/labels.txt',
               inFileImgStr='../assets/gymnasium_playground_Folding-v0.png',
               initX=2,
               initY=2)
env.reset()
env.render()

for i in range(10000):
    ac = env.action_space.sample()
    env.step(ac)
    env.render()
