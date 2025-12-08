from gymnasium import Wrapper
import numpy as np


class DataWrapper(Wrapper):
    
    def __init__(self, env):
        super().__init__(env)
    
    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        reward = self.compute_reward(observation['achieved_goal'], observation['desired_goal'])
        return observation, reward, terminated, truncated, info
    
    @staticmethod
    def compute_reward(achieved_goal, desired_goal):
        d = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
        return -(d > 0.05).astype(np.float32)
    