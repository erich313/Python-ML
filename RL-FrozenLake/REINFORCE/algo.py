import torch
import torch.nn.functional as F

from net import PolicyNet

class REINFORCE:
    def __init__(self, state_dim, hidden_dim, action_dim, learning_rate, gamma):
        self.policy_net = PolicyNet(state_dim, action_dim, hidden_dim)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.gamma = gamma 

    def take_action(self, state, num_states): 
        state = F.one_hot(torch.tensor(state), num_classes=num_states).float().unsqueeze(0)
        # print("State:", state)
        probs = self.policy_net(state)
        # print("Action probabilities:", probs)
        action_dist = torch.distributions.Categorical(probs)
        # print("Action distribution:", action_dist)
        action = action_dist.sample()
        # print("Sampled action:", action)
        return action.item()

    def update(self, transition_dict, num_states):
        reward_list = transition_dict['rewards']
        state_list = transition_dict['states']
        action_list = transition_dict['actions']

        G = 0
        self.optimizer.zero_grad()
        for i in reversed(range(len(reward_list))):
            reward = reward_list[i]
            state = F.one_hot(torch.tensor(state_list[i]), num_classes=num_states).float().unsqueeze(0)
            action = torch.tensor([action_list[i]]).view(-1, 1)
            log_prob = torch.log(self.policy_net(state).gather(1, action))
            G = self.gamma * G + reward
            loss = -log_prob * G
            loss.backward()
        self.optimizer.step()
