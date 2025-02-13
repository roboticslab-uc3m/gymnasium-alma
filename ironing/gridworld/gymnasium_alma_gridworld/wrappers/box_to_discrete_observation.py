import gymnasium as gym
import numpy as np
from gymnasium.spaces import Discrete

class BoxToDiscreteObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.originalRange = (env.observation_space.high - env.observation_space.low)
        self.observation_space = Discrete(self.originalRange[0] * self.originalRange[1])

    def observation(self, obs):
        return obs[1]*self.originalRange[1] + obs[0]
