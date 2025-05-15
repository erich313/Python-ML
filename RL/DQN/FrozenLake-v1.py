import gymnasium as gym
import torch
import torch.nn.functional as F
import itertools
import random
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from dqn import DQN
from experience_replay import ReplayMemory

matplotlib.use('Agg')

class Agent:
    def __init__(self):
        self.replay_memory_size = 1000
        self.batch_size = 32
        self.epsilon_decay = 0.001
        self.epsilon_min = 0.05
        self.network_sync_rate = 10
        self.alpha = 0.001
        self.gamma = 0.9
        self.loss_fn = torch.nn.MSELoss()
        self.optimizer = None

    def run(self, training, render, episodes=1000):
        env = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=False, render_mode=render)

        num_states = env.observation_space.n
        num_actions = env.action_space.n

        epsilon = 1 # 1 = 100% random actions
        memory = ReplayMemory(self.replay_memory_size)

        # Create policy and target network. Number of nodes in the hidden layer can be adjusted.
        policy_dqn = DQN(num_states, num_actions, num_states)
        target_dqn = DQN(num_states, num_actions, num_states)

        # Make the target and policy networks the same (copy weights/biases from one network to the other)
        target_dqn.load_state_dict(policy_dqn.state_dict())

        # Policy network optimizer. "Adam" optimizer can be swapped to something else. 
        self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.alpha)

        # List to keep track of rewards collected per episode. Initialize list to 0's.
        rewards_per_episode = np.zeros(episodes)

        # List to keep track of epsilon decay
        epsilon_history = []

        # Track number of steps taken. Used for syncing policy => target network.
        step_count=0
            
        for episode in range(episodes):
            state = env.reset()[0]  # Initialize to state 0
            state = F.one_hot(torch.tensor(state), num_classes=num_states).float()

            terminated = False      # True when agent falls in hole or reached goal
            truncated = False       # True when agent takes more than 200 actions    

            # Agent navigates map until it falls into hole/reaches goal (terminated), or has taken 200 actions (truncated).
            for t in itertools.count():
                # Select action based on epsilon-greedy
                if random.random() < epsilon:
                    # select random action
                    action = env.action_space.sample() # actions: 0=left,1=down,2=right,3=up
                    action = torch.tensor(action, dtype=torch.int64)
                else:
                    # select best action            
                    with torch.no_grad():
                        action = policy_dqn(state.unsqueeze(dim=0)).squeeze().argmax()

                # Execute action
                new_state,reward,terminated,truncated,_ = env.step(action.item())

                new_state = F.one_hot(torch.tensor(new_state), num_classes=num_states).float()
                reward = torch.tensor(reward, dtype=torch.float)

                # Save experience into memory
                memory.append((state, action, new_state, reward, terminated, truncated)) 

                if terminated or truncated:
                    break

                # Move to the next state
                state = new_state

                # Increment step counter
                step_count+=1

            # Keep track of the rewards collected per episode.
            if reward.item() == 1.0:
                rewards_per_episode[episode] = 1

            # Check if enough experience has been collected and if at least 1 reward has been collected
            if len(memory)>self.batch_size and np.sum(rewards_per_episode)>0:
                batch = memory.sample(self.batch_size)
                self.optimize(batch, policy_dqn, target_dqn)        

                # Decay epsilon
                epsilon = max(epsilon - self.epsilon_decay, self.epsilon_min)
                epsilon_history.append(epsilon)

                # Copy policy network to target network after a certain number of steps
                if step_count > self.network_sync_rate:
                    target_dqn.load_state_dict(policy_dqn.state_dict())
                    step_count=0

        # Close environment
        env.close()

        # Save policy
        torch.save(policy_dqn.state_dict(), "frozen_lake_dql.pt")

        # Create new graph 
        plt.figure(1)

        # Plot average rewards (Y-axis) vs episodes (X-axis)
        sum_rewards = np.zeros(episodes)
        for x in range(episodes):
            sum_rewards[x] = np.sum(rewards_per_episode[max(0, x-100):(x+1)])
        plt.subplot(121) # plot on a 1 row x 2 col grid, at cell 1
        plt.plot(sum_rewards)
        
        # Plot epsilon decay (Y-axis) vs episodes (X-axis)
        plt.subplot(122) # plot on a 1 row x 2 col grid, at cell 2
        plt.plot(epsilon_history)
        
        # Save plots
        plt.savefig('frozen_lake_dql.png')
    
    def optimize(self, batch, policy_dqn, target_dqn):
        state, action, new_state, reward, terminated, truncated = zip(*batch)

        state = torch.stack(state)
        action = torch.stack(action)
        new_state = torch.stack(new_state)
        reward = torch.stack(reward)
        terminated = torch.tensor(terminated).float()
        truncated = torch.tensor(truncated).float()

        with torch.no_grad():
            targetQ = reward + (1-terminated) * (1-truncated) * self.gamma * target_dqn(new_state).max(dim=1)[0]
        
        currentQ = policy_dqn(state).gather(1, action.unsqueeze(dim=1)).squeeze()

        loss = self.loss_fn(currentQ, targetQ)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


if __name__ == "__main__":
    agent = Agent()
    agent.run(training=True, render=None)
