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
        if rand < 1/3:
            # 10% chance to go the first perpendicular way
            action = self.perpandicular[action][0]
        elif rand < 2/3:
            # 10% chance to go the second perpendicular way
            action = self.perpandicular[action][1]
        else:
            # 80% chance to go the original way
            pass
        return self.env.step(action)


def run(episodes, render):

    env = gym.make('FrozenLake-v1', map_name="8x8", is_slippery=False, render_mode='human' if render else None, max_episode_steps=250)
    cenv = CustomEnvWrapper(env)

    f = open('frozen_lake8x8.pkl', 'rb')
    q = pickle.load(f)
    f.close()

    for i in range(8):
        for j in range(8):
            print(f'state{i*8+j}: {q[i*8+j]}')
    return 0

    rewards_per_episode = np.zeros(episodes)

    for i in range(episodes):
        state = cenv.reset()[0] 
        terminated = False      
        truncated = False      

        while(not terminated and not truncated):
            action = np.argmax(q[state,:])
            new_state, reward, terminated, truncated, info= cenv.step(action)
            state = new_state

        if reward == 1:
            rewards_per_episode[i] = 1

    cenv.close()

    sum_rewards = np.zeros(episodes)
    for t in range(episodes):
        sum_rewards[t] = np.sum(rewards_per_episode[max(0, t-100):(t+1)])
    plt.plot(sum_rewards)
    plt.savefig('trained.png')


if __name__ == '__main__':
    # run(1000, render=False)

    run(1, render=True)

