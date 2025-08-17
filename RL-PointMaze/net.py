import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import os


LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
EPSILON = 1e-6


def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

class CriticNet(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, checkpoint_dir='checkpoints', name='CriticNetwork'):
        super(CriticNet, self).__init__()

        # Critic1
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

        # Critic2
        self.fc4 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, hidden_dim)
        self.fc6 = nn.Linear(hidden_dim, 1)

        self.name = name
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, self.name)

        self.apply(weights_init_)
    

    def forward(self, state, action):
        x = torch.cat([state, action], 1)

        x1 = F.relu(self.fc1(x))
        x1 = F.relu(self.fc2(x1))
        x1 = self.fc3(x1)

        x2 = F.relu(self.fc1(x))
        x2 = F.relu(self.fc2(x2))
        x2 = self.fc6(x2)

        return x1, x2
    

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)


    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))


class ActorNet(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, action_space=None, checkpoint_dir='checkpoints', name='ActorNetwork'):
        super(ActorNet, self).__init__()

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean_linear = nn.Linear(hidden_dim, action_dim)
        self.log_std_linear = nn.Linear(hidden_dim, action_dim)

        self.name = name
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, self.name)

        self.apply(weights_init_)

        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor((action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor((action_space.high + action_space.low) / 2.)


    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)

        return mean, log_std
    

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t) - torch.log(self.action_scale * (1 - y_t.pow(2)) + EPSILON)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias

        return action, log_prob, mean


    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(ActorNet, self).to(device)

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)


    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))


class ICMNet(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, checkpoint_dir='checkpoints', name='ICMNetwork'):
        super(ICMNet, self).__init__()

        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, state_dim)

        self.checkpoint_dir = checkpoint_dir
        self.name = name
        self.checkpoint_file = os.path.join(self.checkpoint_dir, self.name)


    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        predictived_state = self.fc3(x)
        return predictived_state
    

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)


    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))
