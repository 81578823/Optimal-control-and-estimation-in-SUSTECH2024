import numpy as np
import os

from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box


DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 0,
    "distance": 2.04,
}


class InvertedPendulumEnv(MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 25,
    }

    def __init__(self, xml_file="inverted_pendulum_project.xml", **kwargs):
        utils.EzPickle.__init__(self, xml_file, **kwargs)
        observation_space = Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float64)
        fullpath = os.path.join(os.path.dirname(__file__), xml_file)
        MujocoEnv.__init__(
            self,
            fullpath,
            2,
            observation_space=observation_space,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            **kwargs,
        )
        self.current_step = 0
        self.max_steps = 500
        self.x_threshold = 10
        self.theta_threshold = 0.2
        self.success_threshold = 0.05
        self.l = 0.6

    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        action = np.clip(action, -3.0, 3.0)[0]
        x, theta = self.data.qpos[0], self.data.qpos[1]
        # x_dot, theta_dot = self.data.qvel[0], self.data.qvel[1]
        
        # 将 theta 转换为 -pi 到 pi 范围内
        theta = ((theta + np.pi) % (2 * np.pi)) - np.pi
        
        if x > self.x_threshold:
            x = self.x_threshold
        elif x < -self.x_threshold:
            x = -self.x_threshold
        
        # reward function as described in dissertation of Deisenroth with A=1
        A = 1
        invT = A * np.array([[1, self.l, 0], [self.l, self.l ** 2, 0], [0, 0, self.l ** 2]])
        j = np.array([x, np.sin(theta), np.cos(theta)])
        j_target = np.array([0.0, 0.0, 1.0])

        reward = np.matmul((j - j_target), invT)
        reward = np.matmul(reward, (j - j_target))
        reward = -(1 - np.exp(-0.5 * reward))
        
        done = False

        self.current_step += 1

        # terminate the game if t >= time limit
        if self.current_step >= self.max_steps:
            done = True

        observation = self._get_obs()
        
        if self.render_mode == "human":
            self.render()
        
        return observation, reward, done, False, {}


    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(
            size=self.model.nq, low=-0.01, high=0.01
        )
        qpos[0] = 0
        qpos[1] = np.pi
        qvel = self.init_qvel + self.np_random.uniform(
            size=self.model.nv
        )
        self.set_state(qpos, qvel)
        self.current_step = 0
        return self._get_obs()

    def _get_obs(self):
        return np.concatenate([self.data.qpos, self.data.qvel]).ravel()
