import gymnasium as gym
import gymnasium_robotics
import numpy as np

from wrapper import DataWrapper
from net import *
from agent import Agent
from buffer import *


if __name__ == "__main__":
    episodes = 5000
    gamma = 0.99
    lambdaa = 0.95
    epsilon = 0.2
    entropy_coef = 0.02
    epochs = 4
    hidden_dim = 512
    learning_rate_ac = 1e-4
    learning_rate_cr = 3e-4
    max_episode_steps = 100
    batch_size = 1024
    mini_batch_size = 256

    STRAIGHT_MAZE = [[1, 1, 1, 1, 1],
                     [1, 0, 0, 0, 1],
                     [1, 1, 1, 1, 1]]
    
    gym.register_envs(gymnasium_robotics)
    env = gym.make('PointMaze_UMaze-v3', max_episode_steps=max_episode_steps, render_mode='human', maze_map=STRAIGHT_MAZE)
    env = DataWrapper(env)

    observation, info = env.reset()

    observation_dim = observation.shape[0]
    action_dim=env.action_space.shape[0]

    agent = Agent(state_dim=observation_dim, 
                  action_space=env.action_space,
                  hidden_dim=hidden_dim,
                  learning_rate_ac=learning_rate_ac,
                  learning_rate_cr=learning_rate_cr,
                  gamma=gamma,
                  lambdaa=lambdaa,
                  epsilon=epsilon,
                  entropy_coef=entropy_coef,
                  epochs=epochs)
    
    buffer = RolloutBuffer(batch_size=batch_size,
                           mini_batch_size=mini_batch_size,
                           state_dim=observation_dim,
                           action_dim=action_dim)

    agent.train(env,
                episodes=episodes,
                summary_writer_name=f'straight_maze',
                max_episode_steps=max_episode_steps,
                buffer=buffer)

    env.close()
    