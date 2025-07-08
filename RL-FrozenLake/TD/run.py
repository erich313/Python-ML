import gymnasium as gym
from gymnasium import Wrapper
import numpy as np
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
        rand = np.random.rand()
        if rand < 0.1:
            # 10% chance to go the first perpendicular way
            action = self.perpandicular[action%2][0]
        elif rand < 0.2:
            # 10% chance to go the second perpendicular way
            action = self.perpandicular[action%2][1]
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


def run(episodes, render):

    env = gym.make('FrozenLake-v1', map_name="8x8", is_slippery=False, render_mode='human' if render else None, max_episode_steps=200)
    cenv = CustomEnvWrapper(env)

    f = open('frozen_lake8x8.pkl', 'rb')
    q = pickle.load(f)
    f.close()

    rewards_per_episode = np.zeros(episodes)


    for i in range(episodes):
        state = cenv.reset()[0] 
        terminated = False      
        truncated = False      

        total_reward = 0
        while(not terminated and not truncated):
            action = np.argmax(q[state,:])
            new_state, reward, terminated, truncated, info= cenv.step(action)
            state = new_state

            total_reward += reward
        
        rewards_per_episode[i] = total_reward

    cenv.close()

    sum_rewards = np.zeros(episodes)
    for t in range(episodes):
        sum_rewards[t] = np.sum(rewards_per_episode[max(0, t-100):(t+1)])/2000
    plt.plot(sum_rewards)
    plt.savefig('trained.png')


if __name__ == '__main__':
    # run(15000, render=False)

    run(1, render=True)

