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
            0: [1, 3],  # remainder = 0: left/right -> down, up
            1: [0, 2],  # remainder = 1: down/up -> left, right
        } 
    
    # def set_state(self, state):
        

    def step(self, action): 
        # Add some stochasticity
        ''' Define the stochasticity

        Original stochasticity:
        General idea: 1/3 for intended action, 2/3 for perpendicular actions
        case 1: 4 valid actions, 1/3 for intended action, 1/3 for one perpendicular action, 1/3 for the other perpendicular action
        case 2: 3 valid actions, 1/3 for intended action, 2/3 for one perpendicular action
        case 3: 2 valid actions, 1/3 for intended action, 2/3 for one other perpendicular action

        Improved stochasticity:
        General idea: 80% for intended action, 20% for perpendicular actions
        case 1: 4 valid actions, 80% for intended action, 10% for one perpendicular action, 10% for the other perpendicular action
        case 2: 3 valid actions, 80% for intended action, 20% for one perpendicular action
        case 3: 2 valid actions, 80% for intended action, 20% for one other perpendicular action    
        '''
        '''
        rand = np.random.rand()
        intersection = list(set(valid_actions) & set(self.perpandicular[action%2]))
        if len(intersection) == 2:
            if rand < 1/3:
                # 10% chance to go the first perpendicular way
                action = self.perpandicular[action%2][0]
            elif rand < 2/3:
                # 10% chance to go the second perpendicular way
                action = self.perpandicular[action%2][1]
            else:
                # 80% chance to go the original way
                pass
            
        elif len(intersection) == 1:
            if rand < 1/3:
                # 20% chance to go the only valid perpendicular way
                action = intersection[0]
            else:
                # 80% chance to go the original way
                pass
        else:
            pass
        '''
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
    
    def valid_action(self, state):
        row, col = state // 8, state % 8
        v = []
        if col > 0: v.append(0)  # col=0, left is not valid
        if row < 7: v.append(1)  # row=7, down is not valid 
        if col < 7: v.append(2)  # col=7, right is not valid
        if row > 0: v.append(3)  # row=0, up is not valid
        return v


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
        state = env.reset()[0] 
        # statearray = []
        # control = 5 - (episode//2000)
        # if control >= 0:
        #     for i in range(control, 8):
        #         for j in range(control, 8):
        #             if custom_map[i][j] == "F" or custom_map[i][j] == "S":
        #                 statearray.append(8*i+j)  
        #     state = np.random.choice(statearray)  # choose a random state from the valid states 
        # else:
        #     # Reset the environment and get the first state
        #     state = env.reset()[0]        

        for t in itertools.count():
            valid_actions = env.valid_action(state)
            # Take a step
            rng = np.random.default_rng()  # random number generator
            if rng.random() < epsilon:
                # actions: 0=left,1=down,2=right,3=up
                # action = env.action_space.sample()  # Exploration: choose a random action 
                action = np.random.choice(valid_actions)  # Exploration: choose a valid random action
            else:
                # action = np.argmax(Q[state])  # Exploitation: choose the best action
                action = np.argmax([Q[state][i] if i in valid_actions else -np.inf for i in range(4)])  # Exploitation: choose the best action
    
            next_state, reward, terminated, truncated, _ = env.step(action)
            # Q-Learning update
            # Q(s,a) <- Q(s,a) + a * [r + gamma * max_a' Q(s',a') - Q(s,a))
            # best_next_action = np.argmax(Q[next_state])
            valid_next_actions = env.valid_action(next_state)
            best_next_action = np.argmax([Q[next_state][i] if i in valid_next_actions else -np.inf for i in range(4)])
            Q[state][action] = Q[state][action] + alpha * (reward + gamma * Q[next_state][best_next_action] - Q[state][action])
            if terminated or truncated:
                break
            
            # Transition to the next state
            state = next_state

        epsilon = max(epsilon - epsilon_decay_rate, 0)
        if(epsilon==0):
            alpha = 0.001
         
        rewards_per_episode[episode] = reward

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
    custom_map=[
        "FFFFFFFF",
        "FFFFFFFF",
        "FFFHFFFF",
        "FFFFFHFF",
        "FFFHSFFF",
        "FHHFFFHF",
        "FHFFHFHF",
        "FFFHFFFG"
    ]
    env = gym.make('FrozenLake-v1', map_name="8x8", is_slippery=False, render_mode=None, max_episode_steps=200, desc=custom_map)
    cenv = CustomEnvWrapper(env)
    Q = Q_Learning(cenv, num_episodes=15000)
