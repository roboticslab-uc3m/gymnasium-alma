import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces

from gymnasium.envs.mujoco import MujocoEnv

import os
import glfw
import mujoco

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


class FoldingEnv(MujocoEnv):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 100,
    }

    def __init__(self, inFileLabelsStr='../assets/labels.txt', inFileImgStr='../assets/gymnasium_playground_FakeFolding-v0.png', initX=2, initY=2, **kwargs):

        # Remember "Coordinate Systems for `.csv` and `print(numpy)`", above.

        self.inFileLabelsStr = inFileLabelsStr
        self.inFileLabels = np.genfromtxt(inFileLabelsStr, delimiter=',')

        self.inFileImg = pygame.image.load(inFileImgStr)
        self.WINDOW_WIDTH = self.inFileImg.get_width()
        self.WINDOW_HEIGHT = self.inFileImg.get_height()

        #self.nS = self.WINDOW_WIDTH * self.WINDOW_HEIGHT  # nS: number of states
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.WINDOW_WIDTH, self.WINDOW_HEIGHT), dtype=int)

        # mujoco model
        self.model_path = "../mujoco_model/scene_position_cloth.xml"
        self.viewer = None
        self.frame_skip = 5


        # assert render_mode is None or render_mode in self.metadata["render_modes"]
        # self.render_mode = render_mode


        MujocoEnv.__init__(
            self,
            os.path.abspath(self.model_path),
            5,
            observation_space = self.observation_space,
            width=int(32),
            height=int(18),
            **kwargs
        )

        self.nA = self.WINDOW_WIDTH * self.WINDOW_HEIGHT # nA: number of actions
        self.action_space = spaces.Discrete(self.nA)

        self.window = None
        self.clock = None

        self._set_data()

        self.init_ctrl = self.data.ctrl[:].copy()

        for j in range(len(self.joints)):
            idx = self.data.actuator(self.joints[j]).id
            self.init_ctrl[idx] = self.home[j]

        for j in range(len(self.joints_gripper)):
            idx = self.data.actuator(self.joints_gripper[j]).id
            self.init_ctrl[idx] = 0.04


        self.init_qpos  = self.data.qpos.copy()
        self.init_qpos[:32] = np.array(self.qpos)
        self.init_qpos[23:30] = np.array(self.rest_qpos)
        self.init_qvel = np.zeros(len(self.data.qvel))
        self.init_qpos[11] = 0.3

    

        # fold
        self.init_qpos[-5] = 1.7
        self.init_qpos[-6] = 3

        self.init_qpos[-15] = 1.7
        self.init_qpos[-16] = 3

        self.init_qpos[-25] = 1.7
        self.init_qpos[-26] = 3

        # rotation
        self.init_qpos[32] = 1 # x position
        self.init_qpos[38] = -0.4 # yaw rotation


    def _set_data(self):
        self.left_home = [-1.1, 1.4679, 2.714, 1.7095, -1.5708, 1.37, 0]
        self.right_home = [0.641, -0.286, 1.204, 2.232, -1.035, -1.379, 0.099]
        self.home = self.left_home + self.right_home
        self.qpos = [0, 0,   -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1.1, 1.4679, 2.714, 1.7095, -1.5708, 1.37, 0, 0, 0, 0.641, -0.286, 1.204, 2.232, -1.035, -1.379, 0.099, 0.04, 0.04]
        self.joints_left = ["arm_left_" + str(i+1) +"_joint_position"  for i in range(7)]
        self.joints_right = ["arm_right_" + str(i+1) +"_joint_position" for i in range(7)]
        self.joints_gripper = ["gripper_right_right_finger_joint_position", 
                        "gripper_right_left_finger_joint_position", 
                        "gripper_left_right_finger_joint_position", 
                        "gripper_left_left_finger_joint_position"]
        self.joints = self.joints_left + self.joints_right
        self.rest_qpos = [0.47807414925570263, -1.1729907312031171, 2.662614000377364, 1.5985649970946019, 1.371503715463566, -1.1855814127187034, -2.156952310975745]
        self.fixed_torso = 0.3
        


    def seed(self, seed=None):
        pass

    def reset_model(self):
        self.current_step = 0

        print(self.model.nq, self.model.nv)
        print(self.init_qpos.shape, self.init_qvel.shape)

        self.set_state(self.init_qpos, self.init_qvel)
        super().do_simulation(self.init_ctrl, 1)

        obs = self._get_obs()
        return obs
    

    def do_simulation(self, ctrl, n_frames) -> None:
        full_ctrl = np.copy(self.data.ctrl[:])
        if len(ctrl) < len(full_ctrl):
            new_ctrl = np.copy(full_ctrl)
            for j in range(len(self.joints)):
                idx = self.data.actuator(self.joints[j]).id
                new_ctrl[idx] = ctrl[j]
            ctrl = new_ctrl

        for j in range(len(self.joints)):
            idx = self.data.actuator(self.joints[j]).id
            full_ctrl[idx] = ctrl[idx]

        full_ctrl[4] = self.fixed_torso

        return super().do_simulation(full_ctrl, n_frames)
    

    def render(self, mode='human'):
        track_img = self.mujoco_renderer.render(self.render_mode, camera_name="track")
        head_img = self.mujoco_renderer.render("rgb_array", camera_name="head_cam")

        imgs = {"track" : track_img,
                "head_cam" : head_img}
        return track_img



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
        #print('FoldingEnv.step', action)

        # candidate_state = self._agent_location + self._action_to_direction[action]
        # try:
        #     candidate_state_tag = self.inFile[candidate_state[0]][candidate_state[1]]
        # except IndexError as e:
        #     # state preserved
        #     print('FoldingEnv.step: full exception message:', e)
        #     print(
        #         'FoldingEnv.step: probably went out of bounds, add some walls on your map!')
        #     terminated = True
        #     quit()

        # if candidate_state_tag == 0:  # void
        #     self._agent_location = candidate_state
        #     reward = -0.5
        #     terminated = True
        # elif candidate_state_tag == 1:  # ok
        #     self._agent_location = candidate_state
        #     reward = 0
        #     terminated = False
        # elif candidate_state_tag == 2:  # pending
        #     self._agent_location = candidate_state
        #     self.inFile[candidate_state[0]][candidate_state[1]] = 1
        #     reward = 0.5
        #     terminated = False
        # else:
        #     print('FoldingEnv.step: found wicked tag, please review!')
        #     terminated = True
        #     quit()

        ctrl = self.data.ctrl[:].copy()
        for j in range(0, 7):
            idx = self.data.actuator(self.joints_right[j]).id
            ctrl[idx] = self.rest_qpos[j]

        self.do_simulation(ctrl,self.frame_skip)
        


        if not np.any(self.inFile == 2):
            print('FoldingEnv.step: done yay!')
            reward = 1.0
            terminated = True

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, False, info

    def _render_text(self):
        viewer = np.copy(self.inFile)  # Force a deep copy for rendering.
        viewer[self._agent_location[0], self._agent_location[1]] = 2
        print(viewer)

    # def _render_pygame(self):

    #     if self.window is None:
    #         pygame.init()
    #         pygame.display.init()
    #         self.window = pygame.display.set_mode((self.WINDOW_WIDTH, self.WINDOW_HEIGHT))
    #         self.window.blit(self.inFileImg, (0,0))

    #     if self.clock is None:
    #         self.clock = pygame.time.Clock()

    #     pygame.event.pump()
    #     pygame.display.update()

    #     # We need to ensure that human-rendering occurs at the predefined framerate.
    #     # The following line will automatically add a delay to keep the
    #     # framerate stable.
    #     self.clock.tick(self.metadata["render_fps"])


    def _get_viewer(self):
        from gymnasium.envs.mujoco.mujoco_rendering import WindowViewer
        viewer = WindowViewer(self.model, self.data)
        return viewer


    def __del__(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def close(self):
        print('FoldingEnv.close')
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
