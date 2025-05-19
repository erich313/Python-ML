import gymnasium as gym
from gymnasium import Wrapper
import numpy as np
from collections import defaultdict
import itertools
import matplotlib.pyplot as plt
import pickle


class CustomEnvWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.perpandicular = {
            0: [1, 3],  # left -> down, up
            1: [0, 2],  # down -> left, right
            2: [1, 3],  # right -> down, up
            3: [0, 2]   # up -> left, right
        }

    def step(self, action):       
        # Add some stochasticity
        rand = np.random.rand()
        if rand < 0.1:
            # 10% chance to go the first perpendicular way
            action = self.perpandicular[action][0]
        elif rand < 0.2:
            # 10% chance to go the second perpendicular way
            action = self.perpandicular[action][1]
        else:
            # 80% chance to go the original way
            pass

        # Actually take a step in the environment
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Modify the reward function
        desc = self.env.unwrapped.desc
        nrow, ncol = self.env.unwrapped.nrow, self.env.unwrapped.ncol
        row, col = np.unravel_index(obs, (nrow, ncol))
        tile = desc[row][col].decode("utf-8")

        if tile == "H":
            reward = -10
        elif tile == "G":
            reward = 20
        else:
            reward = -0.1

        # Return obs(next_state), modified reward, terminated, truncated, and info
        return obs, reward, terminated, truncated, info


def Q_Learning(env, num_episodes, alpha=0.9, gamma=0.9, epsilon=1, epsilon_decay_rate=0.0001):
    # Q_Learning algorithm
    '''
    Args:
        env: Custom Wrapper of Gymnasium environment, redefine probability of is_slippery 
        num_episodes: number of episodes
        alpha: learning rate
        gamma: discount factor
        epsilon: exploration rate        
        epsilon_decay_rate: use decay to balance exploration early and exploitation later
    '''

    rewards_per_episode = np.zeros(num_episodes)

    # Q_table: A dictionary that maps states to action values
    Q = defaultdict(lambda: np.zeros(env.action_space.n))  # action-value function

    for episode in range(num_episodes):
        # Print progress every 100 episodes
        if (episode+1) % 100 == 0:
            print(f"Progress: Episode {episode+1}/{num_episodes}")

        # Reset the environment and get the first state
        state  = env.reset()[0]
        total_reward = 0
        for t in itertools.count():
            # Take a step
            rng = np.random.default_rng()  # random number generator
            if rng.random() < epsilon:
                # actions: 0=left,1=down,2=right,3=up
                action = env.action_space.sample()  # Exploration: choose a random action 
            else:
                action = np.argmax(Q[state])  # Exploitation: choose the best action
            next_state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            
            # Q-Learning update
            # Q(s,a) <- Q(s,a) + a * [r + gamma * max_a' Q(s',a') - Q(s,a))
            best_next_action = np.argmax(Q[next_state])
            Q[state][action] = Q[state][action] + alpha * (reward + gamma * Q[next_state][best_next_action] - Q[state][action])
            if terminated or truncated:
                break
            
            # Transition to the next state
            state = next_state

        # min_epsilon = 0.0
        # reward_threshold = -0.1
        # reward_increment = 0.4
        # reward_target = 180.0
        # steps_to_take = reward_target / reward_increment
        # epsilon_delta = (epsilon - min_epsilon) / steps_to_take

        # if epsilon > min_epsilon and reward >= reward_threshold:
        #     epsilon = max(epsilon - epsilon_delta, min_epsilon)
        #     reward_threshold += reward_increment

        epsilon = max(epsilon - epsilon_decay_rate, 0)
        if(epsilon==0):
            alpha = 0.001
         
        rewards_per_episode[episode] = total_reward

    env.close()

    sum_rewards = np.zeros(num_episodes)
    for t in range(num_episodes):
        sum_rewards[t] = np.sum(rewards_per_episode[max(0, t-100):(t+1)])
    plt.plot(sum_rewards)
    plt.savefig('reward_per_hundred_episodes.png')

    q = np.zeros((env.observation_space.n, env.action_space.n))
    for state, actions in Q.items():
        q[state, :] = actions
    f = open("frozen_lake8x8.pkl","wb")
    pickle.dump(q, f)
    f.close()

    return Q


if __name__ == "__main__":
    env = gym.make('FrozenLake-v1', map_name="8x8", is_slippery=False, render_mode=None, max_episode_steps=100)
    cenv = CustomEnvWrapper(env)
    Q = Q_Learning(cenv, num_episodes=15000)
    