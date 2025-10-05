import torch


class RolloutBuffer:
    def __init__(self, batch_size, mini_batch_size, state_dim, action_dim, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.batch_size = batch_size
        self.device = device
        self.idx = 0
        self.current_size = 0
        self.mini_batch_size = mini_batch_size

        self.states = torch.zeros((batch_size, state_dim), dtype=torch.float32, device=device)
        self.actions = torch.zeros((batch_size, action_dim), dtype=torch.float32, device=device)
        self.rewards = torch.zeros((batch_size, 1), dtype=torch.float32, device=device)
        self.terminateds = torch.zeros((batch_size, 1), dtype=torch.float32, device=device)

    def store(self, state, action, reward, terminated):
        self.states[self.idx] = torch.tensor(state, dtype=torch.float32, device=self.device)
        self.actions[self.idx] = torch.tensor(action, dtype=torch.float32, device=self.device)
        self.rewards[self.idx] = torch.tensor([reward], dtype=torch.float32, device=self.device)
        self.terminateds[self.idx] = torch.tensor([terminated], dtype=torch.float32, device=self.device)

        self.idx = (self.idx + 1) % self.batch_size
        self.current_size = min(self.current_size + 1, self.batch_size)

    def clear(self):
        self.idx = 0
        self.current_size = 0
        self.states.zero_()
        self.actions.zero_()
        self.rewards.zero_()
        self.terminateds.zero_()
