from gymnasium import Wrapper
import numpy as np


class DataWrapper(Wrapper):
    
    def __init__(self, env):
        super().__init__(env)
        self.unwrapped.model.opt.timestep = 0.005

    def reset(self):
        observation, info = self.env.reset()
        observation = self.process_observation(observation)
        return observation, info
    
    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        observation = self.process_observation(observation)
        return observation, reward*2, terminated, truncated, info
    
    def process_observation(self, observation):
        obs_observation = observation['observation']
        obs_achieved_goal = observation['achieved_goal']
        obs_desired_goal = observation['desired_goal']

        obs = np.concatenate((obs_observation, obs_achieved_goal, obs_desired_goal))

        return obs
    