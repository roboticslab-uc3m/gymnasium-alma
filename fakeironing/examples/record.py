#!/usr/bin/env python

import gymnasium as gym
import gymnasium_playground_fakeironing
import pickle

from gymnasium.utils.play import play, PlayPlot

UP = 0
UP_RIGHT = 1
RIGHT = 2
DOWN_RIGHT = 3
DOWN = 4
DOWN_LEFT = 5
LEFT = 6
UP_LEFT = 7

env = gym.make('gymnasium_playground/FakeIroning-v0',
               render_mode='rgb_array',  # "human", "text", None
               inFileStr='../assets/map1.csv',
               initX=2,
               initY=2)

actions = []
storage = []

def callback(obs_t, obs_tp1, action, rew, terminated, truncated, info):
    global actions, storage

    if action != 8:
        actions.append(action)

    if terminated:
        if rew == 1.0:
            storage.append(actions)
            print("[%d] %s (%d)" % (len(storage), actions, len(actions)))

        actions = []

mapping = {
    (ord('w'),): UP,
    (ord('e'),): UP_RIGHT,
    (ord('d'),): RIGHT,
    (ord('c'),): DOWN_RIGHT,
    (ord('x'),): DOWN,
    (ord('z'),): DOWN_LEFT,
    (ord('a'),): LEFT,
    (ord('q'),): UP_LEFT,
}

play(env=env, fps=30, callback=callback, keys_to_action=mapping, noop=8)

with open('actions.pkl', 'wb') as f:
    pickle.dump(storage, f)
