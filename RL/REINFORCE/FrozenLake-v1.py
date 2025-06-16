import gymnasium as gym
import torch
import torch.nn.functional as F
import itertools
import random

from algo import REINFORCE
from pn import PolicyNet
from custom_env import CustomEnvWrapper
from dp import DataProcessor


class Agent:
    def __init__(self):
        # Q-learning parameters
        self.alpha = 0.001
        self.gamma = 0.98


    def run(self, training, render, episodes, slippery):
        # Create environment
        original_env = gym.make('FrozenLake-v1', map_name="4x4", render_mode=render, is_slippery=None, max_episode_steps=200)
        env = CustomEnvWrapper(original_env, slippery=slippery)

        num_states = env.observation_space.n  # observation space, for 4x4 map, 16 states(0-15), for 8x8 map, 64 states(0-63)
        num_actions = env.action_space.n  # action space, 4 actions: 0=left,1=down,2=right,3=up

        # Initialize DataProcessor
        dp = DataProcessor(episodes)

        reinforce = REINFORCE(num_states, 128, num_actions, self.alpha, self.gamma)
            
        for episode in range(episodes):
            state = env.reset()[0]  

            transition_dict = {
                'states': [],
                'actions': [],
                'rewards': []
            }

            terminated = False      # True when agent falls in hole or reached goal
            truncated = False       # True when agent takes more than 100 actions    

            total_reward = 0
            for t in itertools.count():
                action = reinforce.take_action(state, num_states)
                # print(action)

                # The agent take a step
                next_state,reward,terminated,truncated,_ = env.step(action)

                transition_dict['states'].append(state)
                transition_dict['actions'].append(action)
                transition_dict['rewards'].append(reward)

                total_reward += reward

                if terminated or truncated:
                    break

                # Move to the next state
                state = next_state

            reinforce.update(transition_dict, num_states)
            # Track rewards for graphing
            dp.track_rewards(episode, total_reward)

        # Close environment
        env.close()

        if training:
            # Save policy
            # torch.save(policy_dqn.state_dict(), "frozen_lake_dql.pt")

            # Generate a graph
            dp.graphing()


if __name__ == "__main__":
    agent = Agent()
    agent.run(training=True, render=None, episodes=2500, slippery=False)
    # agent.run(training=False, render="human", episodes=5, slippery=True)
