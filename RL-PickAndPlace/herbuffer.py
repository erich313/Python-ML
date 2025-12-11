import numpy as np

class HERReplayBuffer:
    def __init__(self, capacity, obs_dim, goal_dim, action_dim, reward_fn, her_ratio=0.8, max_episode_steps=50):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0
        self.her_ratio = her_ratio
        self.reward_fn = reward_fn
        self.max_episode_steps = max_episode_steps
        
        self.obs_buf = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.ag_buf  = np.zeros((capacity, goal_dim), dtype=np.float32)
        self.dg_buf  = np.zeros((capacity, goal_dim), dtype=np.float32)
        
        self.next_obs_buf = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_ag_buf  = np.zeros((capacity, goal_dim), dtype=np.float32)
        self.next_dg_buf  = np.zeros((capacity, goal_dim), dtype=np.float32)

        self.act_buf = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rew_buf = np.zeros(capacity, dtype=np.float32)
        self.term_buf = np.zeros(capacity, dtype=np.float32)
        self.trunc_buf = np.zeros(capacity, dtype=np.float32)
        

    def store_transition(self, obs_dict, next_obs_dict, action, reward, terminated, truncated):
        idx = self.ptr
        
        self.obs_buf[idx] = obs_dict['observation']
        self.ag_buf[idx]  = obs_dict['achieved_goal']
        self.dg_buf[idx]  = obs_dict['desired_goal']
        
        self.next_obs_buf[idx] = next_obs_dict['observation']
        self.next_ag_buf[idx]  = next_obs_dict['achieved_goal']
        self.next_dg_buf[idx]  = next_obs_dict['desired_goal']
        
        self.act_buf[idx] = action
        self.rew_buf[idx] = reward
        self.term_buf[idx] = terminated
        self.trunc_buf[idx] = truncated
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        

    def can_sample(self, batch_size):
        return self.size >= batch_size

    def sample_buffer(self, batch_size):
        idxs = np.random.randint(0, self.size, size=batch_size)
        
        batch_obs = self.obs_buf[idxs].copy()
        batch_ag  = self.ag_buf[idxs].copy()
        batch_dg  = self.dg_buf[idxs].copy()
        batch_act = self.act_buf[idxs].copy()
        batch_rew = self.rew_buf[idxs].copy()
        batch_term = self.term_buf[idxs].copy()
        batch_trunc = self.trunc_buf[idxs].copy()
        
        batch_next_obs = self.next_obs_buf[idxs].copy()
        batch_next_dg  = self.next_dg_buf[idxs].copy()
        
        her_mask = np.random.rand(batch_size) < self.her_ratio
        
        if her_mask.sum() > 0:
            future_goals = self._sample_future_goals(idxs[her_mask])
            
            batch_dg[her_mask] = future_goals
            batch_next_dg[her_mask] = future_goals
            
            achieved_goals = batch_ag[her_mask]
            new_rewards = self.reward_fn(achieved_goals, future_goals)
            if isinstance(new_rewards, np.ndarray) and new_rewards.ndim > 1:
                new_rewards = new_rewards.flatten()
                
            batch_rew[her_mask] = new_rewards
            
            batch_term[her_mask] = (new_rewards > -0.01).astype(np.float32)

        state_batch = np.concatenate([batch_obs, batch_dg], axis=1)
        next_state_batch = np.concatenate([batch_next_obs, batch_next_dg], axis=1)
        
        return state_batch, next_state_batch, batch_act, batch_rew, batch_term, batch_trunc

    def _sample_future_goals(self, batch_indices):
        new_goals = []
        
        for idx in batch_indices:
            
            steps_to_scan = self.max_episode_steps
            final_transition_idx = idx 
            
            for k in range(steps_to_scan):
                check_idx = (idx + k) % self.capacity
                
                if check_idx == self.ptr and self.size < self.capacity:
                    final_transition_idx = (check_idx - 1 + self.capacity) % self.capacity
                    break
                
                if self.term_buf[check_idx] > 0.5 or self.trunc_buf[check_idx] > 0.5:
                    final_transition_idx = check_idx
                    break
                
                final_transition_idx = check_idx

            if final_transition_idx >= idx:
                dist = final_transition_idx - idx
            else: 
                dist = (self.capacity - idx) + final_transition_idx
            
            offset = np.random.randint(0, dist + 1)
            
            future_idx = (idx + offset) % self.capacity
            new_goals.append(self.ag_buf[future_idx])
            
        return np.array(new_goals)
    