import gymnasium as gym
import numpy as np
from gymnasium import spaces

from time import sleep

import yarp
import roboticslab_kinematics_dynamics as kd

DEFAULT_HEAD_PAN = -45.0
DEFAULT_HEAD_TILT = 0.0

DEFAULT_TRUNK_PAN = 45.41 # 45.0
DEFAULT_TRUNK_TILT = 58.08 # 30.0

"""
# Coordinate Systems for `.csv` and `print(numpy)`

X points down (rows); Y points right (columns); Z would point outwards.

*--> Y (columns: self.inFile.shape[1]; provides the width in pygame)
|
v
X (rows: self.inFile.shape[0]; provides the height in pygame)
"""


class IroningEnv(gym.Env):
    metadata = {"render_modes": ["human", "text"], "render_fps": 4}

    def __init__(self, render_mode=None, inFileStr='map1.csv', initX=2, initY=2):

        #-- Prepare YARP
        yarp.Network.init()
        if not yarp.Network.checkNetwork():
            print('[error] Please try running yarp server')
            quit()

        #-- Prepare Head (H)
        optionsH = yarp.Property()
        optionsH.put('device','remote_controlboard')
        optionsH.put('remote','/teoSim/head')
        optionsH.put('local','/alma/teoSim/head')
        self.ddH = yarp.PolyDriver(optionsH)
        if not self.ddH.isValid():
            print('[error] Cannot connect to: /teoSim/head')
            quit()
        posH = self.ddH.viewIPositionControl()

        #-- Prepare Trunk (T)
        optionsT = yarp.Property()
        optionsT.put('device','remote_controlboard')
        optionsT.put('remote','/teoSim/trunk')
        optionsT.put('local','/alma/teoSim/trunk')
        self.ddT = yarp.PolyDriver(optionsT)
        if not self.ddT.isValid():
            print('[error] Cannot connect to: /teoSim/trunk')
            quit()
        posT = self.ddT.viewIPositionControl()

        #-- Prepare Right Arm (RA)
        optionsRA = yarp.Property()
        optionsRA.put('device','remote_controlboard')
        optionsRA.put('remote','/teoSim/rightArm')
        optionsRA.put('local','/alma/teoSim/rightArm')
        self.ddRA = yarp.PolyDriver(optionsRA)
        if not self.ddRA.isValid():
            print('[error] Cannot connect to: /teoSim/rightArm')
            quit()
        posRA = self.ddRA.viewIPositionControl()
        axesRA = posRA.getAxes()

        #-- Prepare Cartesian Control T and RA (ccTRA)
        optionsCCTRA = yarp.Property()
        optionsCCTRA.put('device', 'CartesianControlClient')
        optionsCCTRA.put('cartesianRemote', '/teoSim/trunkAndRightArm/CartesianControl')
        optionsCCTRA.put('cartesianLocal', '/alma/teoSim/trunkAndRightArm/CartesianControl')
        self.ddccTRA = yarp.PolyDriver(optionsCCTRA)
        if not self.ddccTRA.isValid():
            print('[error] Cannot connect to: /teoSim/trunkAndRightArm/CartesianControl')
            quit()
        self.ccTRA = kd.viewICartesianControl(self.ddccTRA)

        #-- Pre-prog
        posT.positionMove(0, DEFAULT_TRUNK_PAN)
        posT.positionMove(1, DEFAULT_TRUNK_TILT)

        posH.positionMove(0, DEFAULT_HEAD_PAN)
        posH.positionMove(1, DEFAULT_HEAD_TILT)

        for i in range(axesRA):
            posRA.setRefSpeed(i, 25)

        sleep(0.1)
        q = yarp.DVector(axesRA,0.0)
        posRA.positionMove(q)
        while not posRA.checkMotionDone():
            sleep(0.1)

        q = yarp.DVector(axesRA,0.0)
        q[0] = 40
        q[1] = -30
        q[3] = -30
        posRA.positionMove(q)
        while not posRA.checkMotionDone():
            sleep(0.1)

        q = yarp.DVector(axesRA,0.0)
        q[0] = 40
        q[1] = -70
        q[3] = -30
        posRA.positionMove(q)
        while not posRA.checkMotionDone():
            sleep(0.1)

        q = yarp.DVector(axesRA,0.0)
        q[0] = -30
        q[1] = -70
        q[3] = -30
        posRA.positionMove(q)
        while not posRA.checkMotionDone():
            sleep(0.1)

        # 0.6 -0.15 0.05 -0.5 2.52 0.24
        q = yarp.DVector(axesRA,0.0)
        q[0] = -37.8777759318362114982
        q[1] = -75.4999999999999857891
        q[2] = 3.27126779640313225528
        q[3] = -70.8303432830651615859
        q[4] = -88.2557204075698962242
        q[5] = 29.4033175788557450403
        posRA.positionMove(q)
        while not posRA.checkMotionDone():
            sleep(0.1)
        
        sleep(0.2)
        print('> stat')
        x = yarp.DVector()
        ret, state, ts = self.ccTRA.stat(x)
        print('<', yarp.decode(state), '[%s]' % ', '.join(map(str, x)))

        sleep(0.2)
        #xd = yarp.DVector([0.6, -0.15, 0.05, -0.5, 2.52, 0.24])
        #xd = yarp.DVector([0.65, -0.15, 0.0, -0.5, 2.52, 0.24])
        xds = [
            [0.9, -0.17, 0.05, -0.5, 2.52, 0.24],
            [0.9, -0.17, 0.0, -0.5, 2.52, 0.24]
        ]

        for i in range(len(xds)):
            sleep(0.2)
            print('-- movement ' + str(i + 1) + ':')
            xd = yarp.DVector(xds[i])
            print('>', '[%s]' % ', '.join(map(str, xd)))
            if self.ccTRA.movj(xd):
                print('< [ok]')
                print('< [wait...]')
                sleep(0.2)
                ok = self.ccTRA.wait()
                print('> ok', ok)
                print('> stat')
                x = yarp.DVector()
                ret, state, ts = self.ccTRA.stat(x)
                print('<', yarp.decode(state), '[%s]' % ', '.join(map(str, x)))
                self._agent_location_robot = x
            else:
                print('< [fail]')
                quit()

        # Remember "Coordinate Systems for `.csv` and `print(numpy)`", above.

        self.inFileStr = inFileStr
        self.inFile = np.genfromtxt(inFileStr, delimiter=',')

        try:
            self.inFile[initX][initY]
        except IndexError as e:
            print('IroningEnv.__init__: full exception message:', e)
            print('IroningEnv.__init__: init out of bounds, please review code!')
            quit()
        self._initial_agent_location = np.array([initX, initY])

        self.nS = self.inFile.shape[0] * \
            self.inFile.shape[1]  # nS: number of states
        self.observation_space = spaces.Box(
            low=np.array([0, 0]),
            high=np.array([self.inFile.shape[0], self.inFile.shape[1]]), dtype=int)

        self._action_to_direction = {
            0: np.array([-1, 0]),  # UP
            1: np.array([-1, 1]),  # UP_RIGHT
            2: np.array([0, 1]),  # RIGHT
            3: np.array([1, 1]),  # DOWN_RIGHT
            4: np.array([1, 0]),  # DOWN
            5: np.array([1, -1]),  # DOWN_LEFT
            6: np.array([0, -1]),  # LEFT
            7: np.array([-1, -1])  # UP_LEFT
        }
        self._action_to_direction_robot = {
            0: np.array([0, -0.5, 0,   0, 0, 0]),  # UP
            1: np.array([-0.5, -0.5, 0,   0, 0, 0]),  # UP_RIGHT
            2: np.array([-0.5, 0, 0,   0, 0, 0]),  # RIGHT
            3: np.array([-0.5, 0.5, 0,   0, 0, 0]),  # DOWN_RIGHT
            4: np.array([0, 0.5, 0,   0, 0, 0]),  # DOWN
            5: np.array([-0.5, 0.5, 0,   0, 0, 0]),  # DOWN_LEFT
            6: np.array([-0.5, 0, 0,   0, 0, 0]),  # LEFT
            7: np.array([-0.5, -0.5, 0,   0, 0, 0])  # UP_LEFT
        }
        self.nA = 8  # nA: number of actions
        self.action_space = spaces.Discrete(self.nA)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

    def _get_obs(self):
        return self._agent_location

    def _get_info(self):
        return {
            "distance": 0,
            "init": self._agent_location,
            "init_robot": self._agent_location_robot
        }

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random.
        super().reset(seed=seed)

        self.inFile = np.genfromtxt(self.inFileStr, delimiter=',')

        self._agent_location = self._initial_agent_location

        if self.inFile[self._agent_location[0]][self._agent_location[1]] == 2:
            self.inFile[self._agent_location[0]][self._agent_location[1]] = 1

        observation = self._get_obs()
        info = self._get_info()

        self.render()

        return observation, info

    def step(self, action):
        #print('IroningEnv.step', action)

        candidate_state = self._agent_location + self._action_to_direction[action]
        candidate_state = self._agent_location_robot + self._action_to_direction_robot[action]
        try:
            candidate_state_tag = self.inFile[candidate_state[0]][candidate_state[1]]
        except IndexError as e:
            # state preserved
            print('IroningEnv.step: full exception message:', e)
            print(
                'IroningEnv.step: probably went out of bounds, add some walls on your map!')
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
            print('IroningEnv.step: found wicked tag, please review!')
            terminated = True
            quit()

        if not np.any(self.inFile == 2):
            print('IroningEnv.step: done yay!')
            reward = 1.0
            terminated = True

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, False, info

    def render(self):
        #print('IroningEnv.render', self.render_mode)
        if self.render_mode == "human":
            return self._render_pygame()
        if self.render_mode == "text":
            return self._render_text()
        else:  # None
            pass

    def _render_text(self):
        #viewer = np.copy(self.inFile)  # Force a deep copy for rendering.
        #viewer[self._agent_location[0], self._agent_location[1]] = 2
        #print(viewer)
        pass

    def _render_pygame(self):
        pass
        #if self.window is None: # init

        # We need to ensure that human-rendering occurs at the predefined framerate.
        # The following line will automatically add a delay to keep the
        # framerate stable.

    def close(self):
        print('IroningEnv.close')
