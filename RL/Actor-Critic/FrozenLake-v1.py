import gymnasium as gym
import torch
import torch.nn.functional as F
import itertools

from algo import ActorCritic
from custom_env import CustomEnvWrapper
from dp import DataProcessor
from net import PolicyNet


class Agent:
    def __init__(self):
        # Q-learning parameters
        self.gamma = 0.99
        self.lr_actor = 0.001
        self.lr_critic = 0.001
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.device = torch.device("cpu")


    def run(self, render, episodes, slippery):
        # Create environment
        original_env = gym.make('FrozenLake-v1', map_name="8x8", render_mode=render, is_slippery=None, max_episode_steps=200)
        env = CustomEnvWrapper(original_env, slippery=slippery)

        num_states = env.observation_space.n  # observation space, for 4x4 map, 16 states(0-15), for 8x8 map, 64 states(0-63)
        num_actions = env.action_space.n  # action space, 4 actions: 0=left,1=down,2=right,3=up

        # Initialize DataProcessor
        dp = DataProcessor(episodes)

        actorcritic = ActorCritic(num_states, 128, num_actions, self.lr_actor, self.lr_critic, self.gamma, self.device)
            
        for episode in range(episodes):
            state = env.reset()[0] 

            terminated = False      # True when agent falls in hole or reached goal
            truncated = False       # True when agent takes more than 100 actions    

            transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'ends': []}

            total_reward = 0
            for t in itertools.count():

                action = actorcritic.take_action(state, num_states)

                # The agent take a step
                next_state,reward,terminated,truncated,_ = env.step(action)

                transition_dict['states'].append(state)
                transition_dict['actions'].append(action)
                transition_dict['next_states'].append(next_state)
                transition_dict['rewards'].append(reward)
                transition_dict['ends'].append(True if terminated or truncated else False)

                total_reward += reward

                if terminated or truncated:
                    break

                # Move to the next state
                state = next_state

            # Track rewards for graphing
            dp.track_rewards(episode, total_reward)

            actorcritic.update(transition_dict, num_states)

        # Close environment
        env.close()
        torch.save(actorcritic.actor.state_dict(), "frozen_lake_actor-critic.pt")
        dp.graphing()

    def test(self, render, episodes, slippery):
        # Create environment
        original_env = gym.make('FrozenLake-v1', map_name="8x8", render_mode=render, is_slippery=None, max_episode_steps=200)
        env = CustomEnvWrapper(original_env, slippery=slippery)

        num_states = env.observation_space.n  # observation space, for 4x4 map, 16 states(0-15), for 8x8 map, 64 states(0-63)
        num_actions = env.action_space.n  # action space, 4 actions: 0=left,1=down,2=right,3=up

        policynet = PolicyNet(num_states, num_actions, 128)
        # Load the trained model
        policynet.load_state_dict(torch.load("frozen_lake_actor-critic.pt"))

        # Switch the model to evaluation mode
        policynet.eval()

        for episode in range(episodes):
            state = env.reset()[0]  # Initialize to state 0
            state = F.one_hot(torch.tensor(state), num_classes=num_states).float().unsqueeze(dim=0)

            terminated = False      # True when agent falls in hole or reached goal
            truncated = False       # True when agent takes more than 100 actions    

            total_reward = 0
            for t in itertools.count():
                with torch.no_grad():
                    action = policynet(state).squeeze().argmax()

                # The agent take a step
                new_state,reward,terminated,truncated,_ = env.step(action.item())
                new_state = F.one_hot(torch.tensor(new_state), num_classes=num_states).float().unsqueeze(dim=0)
                total_reward += reward

                if terminated or truncated:
                    break

                state = new_state

        # Close environment
        env.close()


if __name__ == "__main__":
    agent = Agent()
    # agent.run(render=None, episodes=2500, slippery=True)
    agent.test(render="human", episodes=5, slippery=True)
