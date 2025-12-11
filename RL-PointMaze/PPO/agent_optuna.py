import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime
import numpy as np 
import optuna 

from net import *
from trick import *
from buffer import *


class Agent(object):
    def __init__(self, state_dim, action_space, hidden_dim, learning_rate_ac, learning_rate_cr, gamma, lambdaa, epsilon, entropy_coef, epochs):
        action_dim = action_space.shape[0]
        
        self.state_dim = state_dim
        self.gamma = gamma
        self.lambdaa = lambdaa
        self.epsilon = epsilon
        self.entropy_coef = entropy_coef
        self.epochs = epochs
        self.learning_rate_ac = learning_rate_ac
        self.learning_rate_cr = learning_rate_cr

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.critic = CriticNet(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.learning_rate_cr, eps=1e-5)


        self.actor = ActorNet(state_dim, action_dim, hidden_dim, action_space).to(self.device)
        self.actor_optim = Adam(self.actor.parameters(), lr=self.learning_rate_ac, eps=1e-5)

        if action_space is None:
            self.action_scale = torch.tensor(1.).to(self.device)
            self.action_bias = torch.tensor(0.).to(self.device)
        else:
            self.action_scale = torch.FloatTensor((action_space.high - action_space.low) / 2.).to(self.device)
            self.action_bias = torch.FloatTensor((action_space.high + action_space.low) / 2.).to(self.device)
    
    
    def select_action(self, state, evaluate=False):
        # ... (This method is UNCHANGED) ...
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        
        action, _, _ = self.actor.sample(state)

        return action.detach().cpu().numpy()[0]
    

    def calculate_gae(self, state_batch, next_state_batch, reward_batch, terminated_batch, gamma, lambdaa):
        rewards = reward_batch.to(self.device)  # (T,1)
        terminated = terminated_batch.to(self.device)  # (T,1)
        
        with torch.no_grad():
            v = self.critic(state_batch).to(self.device) # (T,1)

            last_state = next_state_batch[-1].unsqueeze(0)
            last_value = self.critic(last_state).to(self.device)  # (1,1)
            v = torch.cat([v, last_value], dim=0)  # (T+1,1)

        T = rewards.size(0)
        gae = torch.zeros(T, 1, device=self.device) 
        gae_acc = torch.zeros(1, device=self.device)

        rewards_detach = rewards.detach()
        terminated_detach = terminated.detach()
        v_detach = v.detach()

        for t in reversed(range(T)):
            td_delta = rewards_detach[t] + gamma * v_detach[t + 1] * (1.0 - terminated_detach[t]) - v_detach[t]
            gae_acc = td_delta + gamma * lambdaa * (1.0 - terminated_detach[t]) * gae_acc
            gae[t] = gae_acc

        return gae  
    

    def lr_decay(self, episode, episodes):
        # ... (This method is UNCHANGED) ...
        lr_ac = self.learning_rate_ac * (1 - episode / episodes)
        lr_cr = self.learning_rate_cr * (1 - episode / episodes)
        for i in self.actor_optim.param_groups:
            i['lr'] = lr_ac
        for i in self.critic_optim.param_groups:
            i['lr'] = lr_cr


    def update_parameters(self, buffer: RolloutBuffer):
        state_batch = buffer.states[:buffer.current_size]
        next_state_batch = torch.cat([buffer.states[1:buffer.current_size], buffer.states[buffer.current_size - 1:].clone()])
        action_batch = buffer.actions[:buffer.current_size]
        reward_batch = buffer.rewards[:buffer.current_size]
        terminated_batch = buffer.terminateds[:buffer.current_size]

        gae_batch = self.calculate_gae(state_batch, next_state_batch, reward_batch, terminated_batch, self.gamma, self.lambdaa)
        gae_batch = (gae_batch - gae_batch.mean()) / (gae_batch.std() + 1e-5)

        mean_old, log_std_old = self.actor(state_batch)
        log_std_old = torch.clamp(log_std_old, -20, 2)
        std_old = log_std_old.exp()
        dist_old = torch.distributions.Normal(mean_old.detach(), std_old.detach())

        # Pre-squash the stored action
        z_old = torch.atanh(torch.clamp(
            (action_batch - self.actor.action_bias) / self.actor.action_scale,
            -0.999999, 0.999999
        ))

        # log_prob_old = dist_old.log_prob(z_old)
        # log_prob_old -= torch.log(1 - torch.tanh(z_old).pow(2) + 1e-6)
        # log_prob_old = log_prob_old.sum(dim=1, keepdim=True)

        # --- START FIX (calculating log_prob_old_mb) ---
        log_prob_old = dist_old.log_prob(z_old)
        
        # This is the log-determinant of the Jacobian
        # log(det(da/dz)) = log(scale) + log(1 - tanh(z)^2)
        log_det_jacobian_old = torch.log(self.actor.action_scale) + torch.log(1 - torch.tanh(z_old).pow(2) + 1e-6)
        
        log_prob_old = log_prob_old - log_det_jacobian_old
        log_prob_old = log_prob_old.sum(dim=1, keepdim=True)
        # --- END FIX ---
        
        total_actor_loss = 0
        total_critic_loss = 0
        num_mini_batches = 0

        for _ in range(self.epochs):
            idxs = torch.randperm(buffer.current_size)
            for start in range(0, buffer.current_size, buffer.mini_batch_size):
                end = min(start + buffer.mini_batch_size, buffer.current_size)
                mb_idx = idxs[start:end]

                state_mb = state_batch[mb_idx]
                next_state_mb = next_state_batch[mb_idx]
                action_mb = action_batch[mb_idx]
                log_prob_old_mb = log_prob_old[mb_idx]
                gae_mb = gae_batch[mb_idx]
                reward_mb = reward_batch[mb_idx]
                terminated_mb = terminated_batch[mb_idx]

                # Actor Update
                mean, log_std = self.actor(state_mb)
                log_std = torch.clamp(log_std, -20, 2)
                std = log_std.exp()
                dist = torch.distributions.Normal(mean, std)

                # Pre-squash the stored action
                z = torch.atanh(torch.clamp(
                    (action_mb - self.actor.action_bias) / self.actor.action_scale,
                    -0.999999, 0.999999
                ))

                # log_prob = dist.log_prob(z)
                # log_prob -= torch.log(1 - torch.tanh(z).pow(2) + 1e-6)
                # log_prob = log_prob.sum(dim=1, keepdim=True)

                # --- START FIX (calculating log_prob) ---
                log_prob = dist.log_prob(z)

                # This is the log-determinant of the Jacobian
                log_det_jacobian = torch.log(self.actor.action_scale) + torch.log(1 - torch.tanh(z).pow(2) + 1e-6)
                
                log_prob = log_prob - log_det_jacobian
                log_prob = log_prob.sum(dim=1, keepdim=True)
                # --- END FIX ---

                ratio = torch.exp(log_prob - log_prob_old_mb)
                actor_loss =  - torch.min(ratio * gae_mb, torch.clamp(ratio, 1-self.epsilon, 1+self.epsilon) * gae_mb).mean()

                entropy = dist.entropy().sum(dim=1, keepdim=True)
                actor_loss -= self.entropy_coef * entropy.mean()

                # Critic Update
                with torch.no_grad():
                    td_target = reward_mb + self.gamma * self.critic(next_state_mb).detach() * (1 - terminated_mb)
                critic_loss = F.mse_loss(self.critic(state_mb), td_target)
                    
                self.critic_optim.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.critic_optim.step()

                self.actor_optim.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.actor_optim.step()

                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()

                num_mini_batches += 1

        buffer.clear()

        actor_loss_avg = total_actor_loss / num_mini_batches
        critic_loss_avg = total_critic_loss / num_mini_batches

        return actor_loss_avg, critic_loss_avg 
    

    def train(self, env, episodes, max_episode_steps, buffer: RolloutBuffer, trial: 'optuna.Trial'):
        total_numsteps = 0
        update_count = 0
        
        all_episode_rewards = []

        reward_scaling = RewardScaling(shape=1, gamma=self.gamma)
        state_norm = Normalization(shape=self.state_dim)

        for episode in range(episodes):
            
            true_reward = 0
            episode_reward = 0
            episode_steps = 0
            terminated = False

            state, _ = env.reset()
            
            state = state_norm(state)
            reward_scaling.reset()

            while not terminated and episode_steps < max_episode_steps:
                action = self.select_action(state)
                
                next_state, reward, terminated, _, _ = env.step(action)

                next_state = state_norm(next_state)

                true_reward += reward
                reward = reward_scaling(reward).item()

                episode_steps += 1
                total_numsteps += 1
                episode_reward += reward

                state_t = torch.tensor(state, dtype=torch.float32).to(self.device)
                
                buffer.store(state=state_t, 
                             action=torch.tensor(action, dtype=torch.float32).to(self.device),
                             reward=torch.tensor([reward], dtype=torch.float32).to(self.device),
                             terminated=torch.tensor([terminated], dtype=torch.float32).to(self.device)
                )

                state = next_state

                if buffer.current_size == buffer.batch_size:
                    self.update_parameters(buffer)
                    update_count += 1
                                
            print(f'Episode {episode}, total numsteps: {total_numsteps}, episode steps: {episode_steps}, original reward:{round(true_reward, 2)} scaled reward: {round(episode_reward, 2)}')

            all_episode_rewards.append(true_reward) 

            self.lr_decay(episode, episodes)

            # Report intermediate results every 20 episodes
            if episode > 0 and episode % 20 == 0:
                intermediate_value = np.mean(all_episode_rewards[-20:])
                trial.report(intermediate_value, episode)
                
                # Check if the trial should be pruned
                if trial.should_prune():
                    print(f"Trial {trial.number} pruned at episode {episode}.")
                    raise optuna.exceptions.TrialPruned()

        # Handle case where no episodes were run
        if not all_episode_rewards:
            return -np.inf
        
        # Define the score as average of last 100 episodes and return
        final_avg_reward = np.mean(all_episode_rewards[-100:])
        return final_avg_reward
 