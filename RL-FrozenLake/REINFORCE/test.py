import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

class PolicyNet(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)
        # Initialize weights for better exploration
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1)

class REINFORCE:
    def __init__(self, state_dim, hidden_dim, action_dim, learning_rate, gamma, entropy_beta=0.01):
        self.policy_net = PolicyNet(state_dim, hidden_dim, action_dim)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.entropy_beta = entropy_beta  # Weight for entropy regularization
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net.to(self.device)
        # Moving average baseline
        self.running_return = 0
        self.running_count = 0
        self.alpha = 0.1  # Smoothing factor for moving average

    def take_action(self, state, num_states):
        state = F.one_hot(torch.tensor(state), num_classes=num_states).float().unsqueeze(0).to(self.device)
        probs = self.policy_net(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        entropy = action_dist.entropy()
        return action.item(), log_prob, entropy

    def update(self, transition_dict):
        rewards = transition_dict['rewards']
        log_probs = transition_dict['log_probs']
        entropies = transition_dict['entropies']

        G = 0
        loss = 0
        self.optimizer.zero_grad()

        # Update running return for baseline
        episode_return = sum(rewards)
        self.running_count += 1
        self.running_return = (1 - self.alpha) * self.running_return + self.alpha * episode_return if self.running_count > 1 else episode_return
        baseline = self.running_return

        # Compute policy gradient loss with entropy regularization
        for reward, log_prob, entropy in zip(reversed(rewards), reversed(log_probs), reversed(entropies)):
            G = self.gamma * G + reward
            loss += -log_prob * (G - baseline) - self.entropy_beta * entropy

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

def train_reinforce(episodes=10000, learning_rate=0.005, gamma=0.98, slippery=False):
    # Initialize environment
    env = gym.make('FrozenLake-v1', map_name="8x8", max_episode_steps=100)
    num_states = env.observation_space.n
    num_actions = env.action_space.n

    # Initialize REINFORCE
    agent = REINFORCE(num_states, hidden_dim=128, action_dim=num_actions, 
                      learning_rate=learning_rate, gamma=gamma, entropy_beta=0.01)

    # Track metrics
    rewards = np.zeros(episodes)
    successes = np.zeros(episodes)
    episode_lengths = np.zeros(episodes)
    holes = np.zeros(episodes)

    for episode in range(episodes):
        state = env.reset()[0]
        transition_dict = {'states': [], 'actions': [], 'rewards': [], 'log_probs': [], 'entropies': []}
        total_reward = 0
        success = False
        steps = 0

        while True:
            action, log_prob, entropy = agent.take_action(state, num_states)
            next_state, reward, terminated, truncated, _ = env.step(action)

            # Custom rewards
            desc = env.unwrapped.desc
            row, col = np.unravel_index(next_state, (env.unwrapped.nrow, env.unwrapped.ncol))
            tile = desc[row][col].decode("utf-8")
            if tile == "H":
                reward = -1  # Reduced penalty for holes
                holes[episode] = 1
            elif tile == "G":
                reward = 20
                success = True
            else:
                reward = -0.1

            transition_dict['states'].append(state)
            transition_dict['actions'].append(action)
            transition_dict['rewards'].append(reward)
            transition_dict['log_probs'].append(log_prob)
            transition_dict['entropies'].append(entropy)
            total_reward += reward
            steps += 1

            if terminated or truncated:
                break
            state = next_state

        # Normalize rewards
        rewards_array = np.array(transition_dict['rewards'])
        if rewards_array.std() > 1e-8:
            rewards_array = (rewards_array - rewards_array.mean()) / (rewards_array.std() + 1e-8)
            transition_dict['rewards'] = rewards_array.tolist()

        # Update policy
        agent.update(transition_dict)

        # Track metrics
        rewards[episode] = total_reward
        successes[episode] = 1 if success else 0
        episode_lengths[episode] = steps

        # Log progress
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(rewards[max(0, episode-99):episode+1])
            success_rate = np.mean(successes[max(0, episode-99):episode+1])
            avg_steps = np.mean(episode_lengths[max(0, episode-99):episode+1])
            hole_rate = np.mean(holes[max(0, episode-99):episode+1])
            print(f"Episode {episode+1}/{episodes}, Avg Reward: {avg_reward:.3f}, "
                  f"Success Rate: {success_rate:.3f}, Avg Steps: {avg_steps:.1f}, Hole Rate: {hole_rate:.3f}")

    env.close()

    # Plot results
    plt.figure(figsize=(12, 10))
    window = 100

    plt.subplot(221)
    avg_rewards = [np.mean(rewards[max(0, i-window+1):i+1]) for i in range(episodes)]
    plt.plot(avg_rewards)
    plt.xlabel('Episodes')
    plt.ylabel('Average Reward')
    plt.title('Moving Average Reward')

    plt.subplot(222)
    avg_successes = [np.mean(successes[max(0, i-window+1):i+1]) for i in range(episodes)]
    plt.plot(avg_successes)
    plt.xlabel('Episodes')
    plt.ylabel('Success Rate')
    plt.title('Moving Average Success Rate')

    plt.subplot(223)
    avg_lengths = [np.mean(episode_lengths[max(0, i-window+1):i+1]) for i in range(episodes)]
    plt.plot(avg_lengths)
    plt.xlabel('Episodes')
    plt.ylabel('Average Episode Length')
    plt.title('Moving Average Episode Length')

    plt.subplot(224)
    avg_holes = [np.mean(holes[max(0, i-window+1):i+1]) for i in range(episodes)]
    plt.plot(avg_holes)
    plt.xlabel('Episodes')
    plt.ylabel('Hole Termination Rate')
    plt.title('Moving Average Hole Termination Rate')

    plt.tight_layout()
    plt.savefig('frozen_lake_reinforce_improved.png')
    plt.show()

if __name__ == "__main__":
    train_reinforce(episodes=5000, learning_rate=0.005, gamma=0.98, slippery=False)