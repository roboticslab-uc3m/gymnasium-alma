#!/usr/bin/env python

import gymnasium as gym
import gymnasium_playground_fakefolding

env = gym.make('gymnasium_playground/FakeFolding-v0',
               render_mode='human',  # "human", "text", None
               inFileLabelsStr='../assets/labels.txt',
               inFileImgStr='../assets/gymnasium_playground_FakeFolding-v0.png',
               initX=2,
               initY=2)
env.reset()
env.render()
