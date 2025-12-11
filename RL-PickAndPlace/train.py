import gymnasium as gym
import gymnasium_robotics
import numpy as np

from wrapper import DataWrapper
from net import *
from agent import Agent
from herbuffer import HERReplayBuffer


if __name__ == "__main__":

    replay_buffer_size = 1000000
    episodes = 10000
    batch_size = 256
    update_per_state = 1
    gamma = 0.98
    tau = 0.005
    alpha = 0.2
    target_update_interval = 1
    hidden_dim = 256
    learning_rate = 3e-4
    max_episode_steps = 50
    
    gym.register_envs(gymnasium_robotics)
    env = gym.make('FetchPickAndPlace-v4', max_episode_steps=max_episode_steps)
    env = DataWrapper(env)

    observation_dict, info = env.reset()

    observation_dim = observation_dict['observation'].shape[0]
    goal_dim = observation_dict['desired_goal'].shape[0]
    action_dim=env.action_space.shape[0]

    agent = Agent(state_dim=observation_dim + goal_dim, 
                  action_space=env.action_space,
                  hidden_dim=hidden_dim,
                  gamma=gamma,
                  tau=tau,
                  alpha=alpha,
                  target_update_interval=target_update_interval,
                  learning_rate=learning_rate)

    memory = HERReplayBuffer(
        capacity=replay_buffer_size, 
        obs_dim=observation_dim, 
        goal_dim=goal_dim, 
        action_dim=action_dim, 
        reward_fn=DataWrapper.compute_reward
    )

    agent.train(env,
                memory=memory,
                episodes=episodes,
                batch_size=batch_size,
                updates_per_step=update_per_state,
                summary_writer_name=f'straight_maze',
                max_episode_steps=max_episode_steps)

    env.close()
    