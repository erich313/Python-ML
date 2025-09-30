import gymnasium as gym
import gymnasium_robotics
import numpy as np

from wrapper import DataWrapper
from net import *
from agent import Agent
from buffer import ReplayBuffer


if __name__ == "__main__":

    replay_buffer_size = 1000000
    episodes = 500
    batch_size = 64
    update_per_state = 4
    gamma = 0.99
    tau = 0.005
    alpha = 0.1
    target_update_interval = 1
    hidden_dim = 512
    learning_rate = 1e-4
    max_episode_steps = 100
    exploration_scaling_factor = 1.5

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
                  gamma=gamma,
                  tau=tau,
                  alpha=alpha,
                  target_update_interval=target_update_interval,
                  learning_rate=learning_rate,
                  exploration_scaling_factor=exploration_scaling_factor)

    memory = ReplayBuffer(max_size=replay_buffer_size, state_dim=observation_dim, action_dim=action_dim)

    agent.train(env,
                memory=memory,
                episodes=episodes,
                batch_size=batch_size,
                updates_per_step=update_per_state,
                summary_writer_name=f'straight_maze',
                max_episode_steps=max_episode_steps)

    env.close()
    