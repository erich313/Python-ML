import gymnasium as gym
import torch
import torch.nn.functional as F
import itertools
from tqdm import tqdm

from algo import ActorCritic
from custom_env import CustomEnvWrapper
from dp import DataProcessor  
from net import ActorNet, CriticNet


class Agent:
    def __init__(self):
        # Hyperparameters
        self.gamma = 0.99  # discount factor
        self.lr_actor = 1e-3  # learning rate for actor
        self.lr_critic = 2e-3  # learning rate for critic

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU
        # self.device = torch.device("cpu")  # Use CPU


    def train(self, render, episodes, slippery, max_episode_steps):
        '''
        Train the agent on FrozenLake-v1 environment using Batched Actor-Critic algorithm.
        '''

        # Create environment
        original_env = gym.make('FrozenLake-v1', map_name="8x8", is_slippery=None, max_episode_steps=max_episode_steps, render_mode=render)
        env = CustomEnvWrapper(original_env, slippery=slippery)

        num_states = env.observation_space.n  # observation space, for 4x4 map, 16 states(0-15); for 8x8 map, 64 states(0-63)
        num_actions = env.action_space.n  # action space, 4 actions: 0=left,1=down,2=right,3=up

        # Initialize DataProcessor
        dp = DataProcessor(episodes)

        # Initialize Actor-Critic model
        actorcritic = ActorCritic(num_states, 128, num_actions, self.lr_actor, self.lr_critic, self.gamma, self.device)
            
        # Train the agent
        for episode in tqdm(range(episodes), desc="Training Episodes", ncols=150):
            '''
                Parameters: state, action, next_state, rewards, ends are not one-hot encoded
            '''

            # Initialize to state 0 (starting point)
            state = env.reset()[0] 

            terminated = False  # True when agent falls in hole or reached goal
            truncated = False   # True when agent takes more than max_episode_steps  

            transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'ends': []}

            # Initialize total reward for each episode
            total_reward = 0

            for t in itertools.count():
                # The agent picks an action following the actor
                action = actorcritic.take_action(state, num_states)

                # The agent take a step
                next_state, reward, terminated, truncated, _ = env.step(action)

                transition_dict['states'].append(state)
                transition_dict['actions'].append(action)
                transition_dict['next_states'].append(next_state)
                transition_dict['rewards'].append(reward)
                transition_dict['ends'].append(True if terminated or truncated else False)

                # Keep track of the reward
                total_reward += reward

                if terminated or truncated:
                    break

                # Move to the next state
                state = next_state

            # Update the networks (Batched Actor-Critic Algorithm)
            actorcritic.update(transition_dict, num_states)

            # Track rewards for each episode to graphing
            dp.track_rewards(episode, total_reward)

        # Close environment
        env.close()

        # Save the trained model
        torch.save(actorcritic.actor.state_dict(), "frozen_lake_actor.pt")
        torch.save(actorcritic.critic.state_dict(), "frozen_lake_critic.pt")

        # Graphing the reward per last 100 episodes
        dp.graphing()


    def run(self, render, episodes, slippery, max_episode_steps):
        '''
        Run the agent on the FrozenLake-v1 environment using a trained model.
        '''
        # Create environment
        original_env = gym.make('FrozenLake-v1', map_name="8x8", render_mode=render, is_slippery=None, max_episode_steps=max_episode_steps)
        env = CustomEnvWrapper(original_env, slippery=slippery)

        num_states = env.observation_space.n  # observation space, for 4x4 map, 16 states(0-15); for 8x8 map, 64 states(0-63)
        num_actions = env.action_space.n  # action space, 4 actions: 0=left,1=down,2=right,3=up

        # Initialize the Actor network (Policy network)
        policynet = ActorNet(num_states, num_actions, 128)

        # Load the trained model
        policynet.load_state_dict(torch.load("frozen_lake_actor.pt"))

        # Switch the model to evaluation mode
        policynet.eval()

        # Initialize success list to track success rate
        success = []

        for episode in range(episodes):
            '''
                Parameters: state, next_state are one-hot encoded.
            '''
            
            # Initialize to state 0 (starting point)
            state = env.reset()[0]
            state = F.one_hot(torch.tensor(state), num_classes=num_states).float().unsqueeze(dim=0)

            terminated = False  # True when agent falls in hole or reached goal
            truncated = False   # True when agent takes more than 100 actions    

            for t in itertools.count():
                # Close the gradient tracking
                with torch.no_grad():
                    action = policynet(state).squeeze().argmax()

                # The agent take a step
                next_state, reward, terminated, truncated, _ = env.step(action.item())

                # Keep track of the success data
                if terminated or truncated:
                    if next_state == 63:
                        success.append(1)
                    else:
                        success.append(0)
                    break
                
                # Move to the next state
                next_state = F.one_hot(torch.tensor(next_state), num_classes=num_states).float().unsqueeze(dim=0)
                state = next_state

        # Close environment
        env.close()

        print(f"Success rate over {episodes} episodes: {100 * sum(success) / len(success)} %")

    
    def load(self):
        '''
        Load the Actor network (Policy network) and Critic network (Value network).
        Output the Action Probabilities (tensors) and State Values of each state.
        '''
        num_states = 64
        num_actions = 4

        # Initialize the Actor network (Policy network)
        policynet = ActorNet(num_states, num_actions, 128)

        # Load the trained model
        policynet.load_state_dict(torch.load("frozen_lake_actor.pt"))

        # Switch the model to evaluation mode
        policynet.eval()

        for i in range(num_states):
            state = F.one_hot(torch.tensor(i), num_classes=num_states).float().unsqueeze(dim=0)
            with torch.no_grad():
                action = policynet(state).squeeze()
            print(f"State {i} -> Action {action}")

        # Initialize the Critic network (Value network)
        criticnet = CriticNet(num_states, 128)

        # Load the trained model
        criticnet.load_state_dict(torch.load("frozen_lake_critic.pt"))

        # Switch the model to evaluation mode
        policynet.eval()

        for i in range(num_states):
            state = F.one_hot(torch.tensor(i), num_classes=num_states).float().unsqueeze(dim=0)
            with torch.no_grad():
                value = criticnet(state).squeeze()
            print(f"State {i} -> Value {value.item()}")


if __name__ == "__main__":
    agent = Agent()
    agent.train(render=None, episodes=5000, slippery=True, max_episode_steps=200)
    # agent.run(render='human', episodes=5, slippery=True, max_episode_steps=200)
    # agent.run(render=None, episodes=1000, slippery=True, max_episode_steps=200)
    # agent.load()
