import numpy as np


class ReplayBuffer():

    def __init__(self, max_size, state_dim, action_dim):
        self.mem_size = max_size
        self.mem_counter = 0
        self.state_memory = np.zeros((self.mem_size, state_dim))
        self.next_state_memory = np.zeros((self.mem_size, state_dim))
        self.action_memory = np.zeros((self.mem_size, action_dim))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminated_memory = np.zeros(self.mem_size, dtype=np.bool)
        self.truncated_memory = np.zeros(self.mem_size, dtype=np.bool)
        

    def can_smaple(self, batch_size):
        if self.mem_counter > (batch_size*5):
            return True
        else:
            return False
        
    
    def store_transition(self, state, next_state, action, reward, terminated, truncated):
        index = self.mem_counter % self.mem_size
        self.state_memory[index] = state
        self.next_state_memory[index] = next_state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminated_memory[index] = terminated
        self.truncated_memory[index] = truncated

        self.mem_counter += 1


    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_counter, self.mem_size)
        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        next_states = self.next_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        terminateds = self.terminated_memory[batch]
        truncateds = self.truncated_memory[batch]

        return states, next_states, actions, rewards, terminateds, truncateds
    