import gymnasium as gym
import gymnasium_robotics
import numpy as np

from wrapper import DataWrapper
from net import *
from agent import Agent


if __name__ == "__main__":
    episodes = 10
    gamma = 0.982
    lambdaa = 0.958
    epsilon = 0.12
    entropy_coef = 0.06
    epochs = 10
    hidden_dim = 512
    learning_rate_ac = 2e-4
    learning_rate_cr = 9e-4
    max_episode_steps = 100
    batch_size = 4096
    mini_batch_size = 512

    STRAIGHT_MAZE = [[1, 1, 1, 1, 1],
                     [1, 0, 0, 0, 1],
                     [1, 1, 1, 1, 1]]
    
    gym.register_envs(gymnasium_robotics)
    env = gym.make('PointMaze_UMaze-v3', max_episode_steps=max_episode_steps, render_mode='human', maze_map=STRAIGHT_MAZE, continuing_task=False)
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

    agent.load_checkpoint(evaluate=True)

    agent.test(env, episodes=episodes, max_episode_steps=max_episode_steps)

    env.close()
