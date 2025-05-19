import gymnasium as gym
import torch
import torch.nn.functional as F
import itertools
import random

from dqn import DQN
from experience_replay import ReplayMemory
from dp import DataProcessor
from custom_env import CustomEnvWrapper


class Agent:
    def __init__(self):
        # Q-learning parameters
        self.alpha = 0.001
        self.gamma = 0.9

        # Epsilon-Greedy Algorithm parameters
        self.epsilon_decay = 0.0001
        self.epsilon_min = 0

        # Experience Replay parameters
        self.replay_memory_size = 5000
        self.batch_size = 64
        self.network_sync_rate = 500

        # Neural Network parasmeters
        self.loss_fn = torch.nn.MSELoss()
        self.optimizer = None


    def run(self, training, render, episodes, slippery):
        # Create environment
        original_env = gym.make('FrozenLake-v1', map_name="8x8", render_mode=render, is_slippery=None, max_episode_steps=100)
        env = CustomEnvWrapper(original_env, slippery=slippery)

        num_states = env.observation_space.n  # observation space, for 4x4 map, 16 states(0-15), for 8x8 map, 64 states(0-63)
        num_actions = env.action_space.n  # action space, 4 actions: 0=left,1=down,2=right,3=up

        # Initialize DataProcessor
        dp = DataProcessor(episodes)

        # Create policy network
        policy_dqn = DQN(num_states, num_actions, 128)

        if training:
            epsilon = 1 

            # Initialize Replay Memory
            memory = ReplayMemory(self.replay_memory_size)

            # Create target network
            target_dqn = DQN(num_states, num_actions, 128)

            # Make the target and policy networks the same (copy weights/biases)
            target_dqn.load_state_dict(policy_dqn.state_dict())

            # Initialize Policy network optimizer
            self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.alpha)

            step_count=0
        else:
            # Load the trained model
            policy_dqn.load_state_dict(torch.load("frozen_lake_dql.pt"))

            # Switch the model to evaluation mode
            policy_dqn.eval()
            
        for episode in range(episodes):
            state = env.reset()[0]  # Initialize to state 0
            state = F.one_hot(torch.tensor(state), num_classes=num_states).float()

            terminated = False      # True when agent falls in hole or reached goal
            truncated = False       # True when agent takes more than 100 actions    

            total_reward = 0
            for t in itertools.count():
                # Select action based on epsilon-greedy algorithm
                if training and random.random() < epsilon:
                    # select random action
                    action = env.action_space.sample() 
                    action = torch.tensor(action, dtype=torch.int64)
                else:
                    # select best action            
                    with torch.no_grad():
                        action = policy_dqn(state.unsqueeze(dim=0)).squeeze().argmax()

                # The agent take a step
                new_state,reward,terminated,truncated,_ = env.step(action.item())
                new_state = F.one_hot(torch.tensor(new_state), num_classes=num_states).float()
                reward = torch.tensor(reward, dtype=torch.float)
                total_reward += reward.item()

                if training:
                    # Save experience into memory
                    memory.append((state, action, new_state, reward, terminated, truncated)) 

                    # Increment step counter
                    step_count += 1

                if terminated or truncated:
                    break

                # Move to the next state
                state = new_state

            # Track rewards for graphing
            dp.track_rewards(episode, total_reward)

            if training:
                # Check if enough experience has been collected and if at least 1 reward has been collected
                if len(memory)>self.batch_size:
                    batch = memory.sample(self.batch_size)
                    self.optimize(batch, policy_dqn, target_dqn)        

                    # Epsilon decay
                    epsilon = max(epsilon - self.epsilon_decay, self.epsilon_min)
                    
                    # Track epsilon for graphing
                    dp.track_epsilon(epsilon)

                    # Copy policy network to target network
                    if step_count > self.network_sync_rate:
                        target_dqn.load_state_dict(policy_dqn.state_dict())
                        step_count=0

        # Close environment
        env.close()

        if training:
            # Save policy
            torch.save(policy_dqn.state_dict(), "frozen_lake_dql.pt")

            # Generate a graph
            dp.graphing()
    

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
    agent.run(training=True, render=None, episodes=15000, slippery=True)
    # agent.run(training=False, render="human", episodes=5, slippery=True)
