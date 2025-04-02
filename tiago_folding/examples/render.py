#!/usr/bin/env python

import gymnasium as gym
import gymnasium_playground_tiago_folding

env = gym.make('gymnasium_playground/TiagoFolding-v0',
               render_mode='human',  # "human", "text", None
               inFileLabelsStr='../assets/labels.txt',
               inFileImgStr='../assets/gymnasium_playground_TiagoFolding-v0.png',
               initX=2,
               initY=2)
ob, _ =env.reset()
env.render()

for i in range(10000):
    env.step(ob)
    env.render()
