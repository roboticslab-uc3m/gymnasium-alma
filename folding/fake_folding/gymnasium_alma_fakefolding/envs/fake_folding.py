import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces

"""
# Coordinate Systems for `.csv` and `print(numpy)`

X points down (rows); Y points right (columns); Z would point outwards.

*--> Y (columns: self.inFile.shape[1]; provides the width in pygame)
|
v
X (rows: self.inFile.shape[0]; provides the height in pygame)
"""

COLOR_BACKGROUND = (0, 0, 0)
COLOR_OK = (255, 255, 255)
COLOR_PENDING = (0, 0, 255)
COLOR_ROBOT = (255, 0, 0)


class FakeFoldingEnv(gym.Env):
    metadata = {"render_modes": ["human", "text"], "render_fps": 4}

    def __init__(self, render_mode=None, inFileLabelsStr='../assets/labels.txt', inFileImgStr='../assets/gymnasium_alma_FakeFolding-v0.png', initX=2, initY=2):

        # Remember "Coordinate Systems for `.csv` and `print(numpy)`", above.

        self.inFileLabelsStr = inFileLabelsStr
        self.inFileLabels = np.genfromtxt(inFileLabelsStr, delimiter=',')

        self.inFileImg = pygame.image.load(inFileImgStr)
        self.WINDOW_WIDTH = self.inFileImg.get_width()
        self.WINDOW_HEIGHT = self.inFileImg.get_height()

        #self.nS = self.WINDOW_WIDTH * self.WINDOW_HEIGHT  # nS: number of states
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.WINDOW_WIDTH, self.WINDOW_HEIGHT), dtype=int)


        self.nA = self.WINDOW_WIDTH * self.WINDOW_HEIGHT # nA: number of actions
        self.action_space = spaces.Discrete(self.nA)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window = None
        self.clock = None

    def _get_obs(self):
        return np.zeros((self.WINDOW_WIDTH, self.WINDOW_HEIGHT), dtype=int)

    def _get_info(self):
        return {
            "distance": 0
        }

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random.
        super().reset(seed=seed)

        self.inFile = np.genfromtxt(self.inFileLabelsStr, delimiter=',')

        observation = self._get_obs()
        info = self._get_info()

        self.render()

        return observation, info

    def step(self, action):
        #print('FakeFoldingEnv.step', action)

        candidate_state = self._agent_location + self._action_to_direction[action]
        try:
            candidate_state_tag = self.inFile[candidate_state[0]][candidate_state[1]]
        except IndexError as e:
            # state preserved
            print('FakeFoldingEnv.step: full exception message:', e)
            print(
                'FakeFoldingEnv.step: probably went out of bounds, add some walls on your map!')
            terminated = True
            quit()

        if candidate_state_tag == 0:  # void
            self._agent_location = candidate_state
            reward = -0.5
            terminated = True
        elif candidate_state_tag == 1:  # ok
            self._agent_location = candidate_state
            reward = 0
            terminated = False
        elif candidate_state_tag == 2:  # pending
            self._agent_location = candidate_state
            self.inFile[candidate_state[0]][candidate_state[1]] = 1
            reward = 0.5
            terminated = False
        else:
            print('FakeFoldingEnv.step: found wicked tag, please review!')
            terminated = True
            quit()

        if not np.any(self.inFile == 2):
            print('FakeFoldingEnv.step: done yay!')
            reward = 1.0
            terminated = True

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, False, info

    def render(self):
        #print('FakeFoldingEnv.render', self.render_mode)
        if self.render_mode == "human":
            return self._render_pygame()
        if self.render_mode == "text":
            return self._render_text()
        else:  # None
            pass

    def _render_text(self):
        viewer = np.copy(self.inFile)  # Force a deep copy for rendering.
        viewer[self._agent_location[0], self._agent_location[1]] = 2
        print(viewer)

    def _render_pygame(self):

        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.WINDOW_WIDTH, self.WINDOW_HEIGHT))
            self.window.blit(self.inFileImg, (0,0))

        if self.clock is None:
            self.clock = pygame.time.Clock()

        pygame.event.pump()
        pygame.display.update()

        # We need to ensure that human-rendering occurs at the predefined framerate.
        # The following line will automatically add a delay to keep the
        # framerate stable.
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        print('FakeFoldingEnv.close')
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
