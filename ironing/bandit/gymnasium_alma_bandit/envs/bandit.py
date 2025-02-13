import gymnasium as gym
import numpy as np
from gymnasium import spaces

class BanditEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, render_mode=None, n_arms=10):
        super(BanditEnv, self).__init__()
        self.observation_space = spaces.Discrete(1)
        self.observation = 0
        self.action_space = spaces.Discrete(n_arms)

    def _get_obs(self):
        return self.observation

    def _get_info(self):
        return {
            "optimal": self.optimal
        }

    def reset(self, seed=None, options=None):
        self.means = np.random.randn(self.action_space.n)
        print({"means": self.means})
        self.optimal = np.argmax(self.means)
        return self._get_obs(), self._get_info()

    def step(self, action):
        assert self.action_space.contains(action)
        reward = (np.random.randn(1) + self.means[action])[0]
        return self._get_obs(), reward, False, False, self._get_info()

    def render(self):
        pass

    def close(self):
        print('BanditEnv.close')
