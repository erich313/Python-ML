import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime
import numpy as np

from net import *


class Agent(object):
    def __init__(self, state_dim, action_space, hidden_dim, learning_rate, gamma, lambdaa, epsilon, epochs):
        action_dim = action_space.shape[0]
        
        self.gamma = gamma
        self.lambdaa = lambdaa
        self.epsilon = epsilon
        self.epochs = epochs

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.critic = CriticNet(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=learning_rate)


        self.actor = ActorNet(state_dim, action_dim, hidden_dim, action_space).to(self.device)
        self.actor_optim = Adam(self.actor.parameters(), lr=learning_rate)

        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor((action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor((action_space.high + action_space.low) / 2.)

    
    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        
        action, _, _ = self.actor.sample(state)

        return action.detach().cpu().numpy()[0]


    def update_parameters(self, transition_dict, gaes):
        state_batch, next_state_batch, action_batch, reward_batch, terminated_batch, gae_batch = transition_dict['states'], transition_dict['next_states'], transition_dict['actions'], transition_dict['rewards'], transition_dict['ends'], gaes

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        terminated_batch = torch.FloatTensor(terminated_batch).to(self.device).unsqueeze(1)
        gae_batch = torch.FloatTensor(gae_batch).to(self.device).unsqueeze(1)
        gae_batch = (gae_batch - gae_batch.mean()) / (gae_batch.std() + 1e-8)

        gae_batch = gae_batch.detach()

        mean_old, log_std_old = self.actor(state_batch)
        log_std_old = torch.clamp(log_std_old, -20, 2)
        std_old = log_std_old.exp()
        dist_old = torch.distributions.Normal(mean_old.detach(), std_old.detach())

        # Pre-squash the stored action
        z_old = torch.atanh(torch.clamp(
            (action_batch - self.actor.action_bias) / self.actor.action_scale,
            -0.999999, 0.999999
        ))

        log_prob_old = dist_old.log_prob(z_old)
        log_prob_old -= torch.log(1 - torch.tanh(z_old).pow(2) + 1e-6)
        log_prob_old = log_prob_old.sum(dim=1, keepdim=True)


        for _ in range(self.epochs):
            # Actor Update
            mean, log_std = self.actor(state_batch)
            log_std = torch.clamp(log_std, -20, 2)
            std = log_std.exp()
            dist = torch.distributions.Normal(mean, std)

            # Pre-squash the stored action
            z = torch.atanh(torch.clamp(
                (action_batch - self.actor.action_bias) / self.actor.action_scale,
                -0.999999, 0.999999
            ))

            log_prob = dist.log_prob(z)
            log_prob -= torch.log(1 - torch.tanh(z).pow(2) + 1e-6)
            log_prob = log_prob.sum(dim=1, keepdim=True)

            ratio = torch.exp(log_prob - log_prob_old)
            actor_loss =  - torch.min(ratio * gae_batch, torch.clamp(ratio, 1-self.epsilon, 1+self.epsilon) * gae_batch).mean()
            entropy = dist.entropy().sum(dim=1, keepdim=True)
            entropy_coef = 0.01
            actor_loss = actor_loss - entropy_coef * entropy.mean()

            # Critic Update
            td_target = gae_batch + self.critic(state_batch).detach()
            critic_loss = F.mse_loss(self.critic(state_batch), td_target.detach())
                
            self.critic_optim.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
            self.critic_optim.step()

            self.actor_optim.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            self.actor_optim.step()

        return actor_loss.item(), critic_loss.item()    
    

    def train(self, env, episodes, summary_writer_name, max_episode_steps):
        warm_up = 20

        summary_writer_name = f'runs/{datetime.now().strftime("%Y%m%d-%H%M%S")}-{summary_writer_name}'
        writer = SummaryWriter(summary_writer_name)

        total_numsteps = 0

        for episode in range(episodes):
            transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'ends': []}

            episode_reward = 0
            episode_steps = 0
            terminated = False
            truncated = False

            rewards = []
            state_value_functions = []

            state, _ = env.reset()

            while not terminated and episode_steps < max_episode_steps:
                action = self.select_action(state)

                # if memory.can_smaple(batch_size=batch_size):
                #     for _ in range(updates_per_step):
                #         critic1_loss, critic2_loss, actor_loss, ent_loss, prediction_loss, alpha = self.update_parameters(memory, batch_size, updates)
                #         writer.add_scalar('Loss/Critic1', critic1_loss, updates)
                #         writer.add_scalar('Loss/Critic2', critic2_loss, updates)
                #         writer.add_scalar('Loss/Actor', actor_loss, updates)
                #         writer.add_scalar('Loss/Entropy', ent_loss, updates)
                #         writer.add_scalar('Loss/Prediction', prediction_loss, updates)
                #         writer.add_scalar('Parameters/Alpha', alpha, updates)
                #         updates += 1
                
                next_state, reward, terminated, truncated, _ = env.step(action)

                episode_steps += 1
                total_numsteps += 1
                episode_reward += reward

                # For Compute Generalized Advantage Estimation (GAE)
                state_tensor = torch.FloatTensor(state).to(self.device).unsqueeze(0)
                state_value_functions.append(self.critic(state_tensor).detach())
                rewards.append(reward)


                transition_dict['states'].append(state)
                transition_dict['actions'].append(action)
                transition_dict['next_states'].append(next_state)
                transition_dict['rewards'].append(reward)
                transition_dict['ends'].append(terminated)

                state = next_state

            writer.add_scalar('Reward/Train', episode_reward, episode)
            print(f'Episode {episode}, total numsteps: {total_numsteps}, episode steps: {episode_steps}, episode reward: {round(episode_reward, 2)}')

            # Compute Generalized Advantage Estimation (GAE)
            final_state_tensor = torch.FloatTensor(next_state).to(self.device).unsqueeze(0)
            state_value_functions.append(self.critic(final_state_tensor).detach())
            gaes = []
            td_deltas = [rewards[t] + self.gamma * state_value_functions[t + 1] - state_value_functions[t] for t in range(len(rewards))]
            T = len(td_deltas)
            for t in range(T):
                advantage = 0
                for i in range(t, T):
                    advantage += (self.gamma * self.lambdaa) ** (i - t) * td_deltas[i]
                gaes.append(advantage)
                


            actor_loss, critic_loss = self.update_parameters(transition_dict, gaes)

            if episode % 10 == 0:
                self.save_checkpoint()

    
    def test(self, env, episodes=10, max_episode_steps=500):
        for episode in range(episodes):
            episode_reward = 0
            episode_steps = 0
            terminated = False
            truncated = False

            state, _ = env.reset()

            while not terminated and episode_steps < max_episode_steps:
                action = self.select_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)

                episode_steps += 1

                if reward == 1:
                    terminated = True

                episode_reward += reward

                state = next_state

            print(f'Test Episode {episode}, steps: {episode_steps}, reward: {round(episode_reward, 2)}')

    def save_checkpoint(self):
        if not os.path.exists('checkpoints/'):
            os.makedirs('checkpoints/')
        
        print('Saving models')

        self.actor.save_checkpoint()
        self.critic.save_checkpoint()


    def load_checkpoint(self, evaluate=False):
        try:
            print("Loading checkpoints...")
            self.actor.load_checkpoint()
            self.critic.load_checkpoint()
            print("Checkpoints loaded successfully.")
        except:
            if evaluate:
                raise Exception("No checkpoints found for evaluation.")
            else:
                print("No checkpoints found, starting from scratch.")
        
        if evaluate:
            self.actor.eval()
            self.critic.eval()
        else:
            self.actor.train()
            self.critic.train()
