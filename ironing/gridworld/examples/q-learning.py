#!/usr/bin/env python
# https://towardsdatascience.com/reinforcement-learning-with-openai-d445c2c687d2

import termios
import tty
import sys
import gymnasium as gym
import gymnasium_alma_gridworld
from gymnasium_alma_gridworld.wrappers import BoxToDiscreteObservation

import numpy as np

env_raw = gym.make('gymnasium_alma/GridWorld-v0',
                   render_mode='human',  # "human", "text", None
                   inFileStr='../assets/map1.csv',
                   initX=2,
                   initY=2,
                   goalX=7,
                   goalY=2)
env = BoxToDiscreteObservation(env_raw)
env.reset()
env.render()

# 1. Create Q-table structure
Q = np.zeros([env.observation_space.n, env.action_space.n])

# 2. Parameters of Q-leanring
eta = .628
gma = .9
epis = 100  # IMPORTANT: increase this value if goal is not being reached
rev_list = []  # rewards per episode calculate

# 3. Q-learning Algorithm
print("Computing Q-Table...")
for i in range(epis):
    # Reset environment
    s, _ = env.reset()
    rAll = 0
    d = False
    j = 0
    # The Q-Table learning algorithm
    while j < 99:
        j += 1
        # Choose action from Q table
        a = np.argmax(Q[s, :] + np.random.randn(1,
                      env.action_space.n)*(1./(i+1)))
        # Get new state & reward from environment
        # print("a",a)
        s1, r, d, _, _ = env.step(a)
        # Update Q-Table with new knowledge
        Q[s, a] = Q[s, a] + eta*(r + gma*np.max(Q[s1, :]) - Q[s, a])
        rAll += r
        s = s1
        # env.render() # comment toggle to see each training step (slower)
        if d == True:
            break
    rev_list.append(rAll)
    env.render()  # comment toggle to see end of each epi
    print("epi", i, "of", epis, "got rAll", rAll)
print("Reward Sum on all episodes " + str(sum(rev_list)/epis))
print("Final Values Q-Table")
print(Q)

print("Press any key to run solution...")
# https://stackoverflow.com/questions/510357/python-read-a-single-character-from-the-user
fd = sys.stdin.fileno()
old_settings = termios.tcgetattr(fd)
try:
    tty.setraw(sys.stdin.fileno())
    ch = sys.stdin.read(1)
finally:
    termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

# Reset environment
s, _ = env.reset()
d = False
# The Q-Table learning algorithm
while d != True:
    # Choose action from Q table
    a = np.argmax(Q[s, :] + np.random.randn(1, env.action_space.n)*(1./(i+1)))
    # Get new state & reward from environment
    s1, r, d, _, _ = env.step(a)
    env.render()
    # Update Q-Table with new knowledge
    Q[s, a] = Q[s, a] + eta*(r + gma*np.max(Q[s1, :]) - Q[s, a])
    s = s1
# Code will stop at d == True, and render one state before it
