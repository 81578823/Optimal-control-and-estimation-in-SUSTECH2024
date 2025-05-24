import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box

class CustomWrapper(gym.Wrapper):
    def __init__(self, env, max_steps=200):
        super().__init__(env)
        self.action_space = Box(low=-3.0, high=3.0, shape=(1,), dtype=np.float32)
        self.max_steps = max_steps
        self.current_step = 0

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        x, theta = observation[0], observation[1]
        x_dot, theta_dot = observation[2], observation[3]

        survival_reward = 5
        angle_reward = np.cos(theta)
        theta_threshold = 0.2
        threshold = 2.0
        if abs(theta) > theta_threshold:
            angle_reward = -50  # 大的负reward表示失败

        if abs(x) <= threshold:
            position_reward = np.exp(-x**2)
        else:
            position_reward = -np.exp(abs(x) - threshold)
        
        velocity_penalty = 2 * (x_dot**2 + theta_dot**2)
        
        custom_reward = survival_reward + angle_reward + position_reward - velocity_penalty
        return observation, custom_reward, terminated, False, {}




