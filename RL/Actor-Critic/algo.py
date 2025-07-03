import torch
import torch.nn.functional as F

from net import PolicyNet, ValueNet


class ActorCritic:
    def __init__(self, state_dim, hidden_dim, action_dim, lr_actor, lr_critic, gamma, device):
        # networks
        self.actor = PolicyNet(state_dim, action_dim, hidden_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)

        # optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)

        # hyperparameters
        self.gamma = gamma
        self.device = device


    def take_action(self, state, num_states):
        state = F.one_hot(torch.tensor(state), num_classes=num_states).float().unsqueeze(0).to(self.device)
        probs = self.actor(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()


    def update(self, transition_dict, num_states):
        states = F.one_hot(torch.tensor(transition_dict['states']), num_classes=num_states).float().to(self.device)
        actions = torch.tensor(transition_dict['actions'], dtype=torch.int64).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = F.one_hot(torch.tensor(transition_dict['next_states']), num_classes=num_states).float().to(self.device)
        ends = torch.tensor(transition_dict['ends'], dtype=torch.float).view(-1, 1).to(self.device)

        td_target = rewards + self.gamma * self.critic(next_states) * (1 - ends)
        td_delta = td_target - self.critic(states)

        log_probs = torch.log(self.actor(states).gather(1, actions))

        actor_loss = torch.mean(-log_probs * td_delta.detach())
        critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach()))

        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        self.actor_optimizer.step()
        self.critic_optimizer.step()

