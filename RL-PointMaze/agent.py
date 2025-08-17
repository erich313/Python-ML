import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime

from net import *
from buffer import ReplayBuffer


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


class Agent(object):
    def __init__(self, state_dim, action_space, hidden_dim, gamma, tau, alpha, target_update_interval, learning_rate, exploration_scaling_factor):
        action_dim = action_space.shape[0]
        
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha

        self.target_update_interval = target_update_interval

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.critic = CriticNet(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=learning_rate)

        self.critic_target = CriticNet(state_dim, action_dim, hidden_dim).to(self.device)
        hard_update(self.critic_target, self.critic)

        self.actor = ActorNet(state_dim, action_dim, hidden_dim, action_space).to(self.device)
        self.actor_optim = Adam(self.actor.parameters(), lr=learning_rate)

        self.icm = ICMNet(state_dim, action_dim, hidden_dim).to(self.device)
        self.icm_optim = Adam(self.icm.parameters(), lr=learning_rate)

        self.exploration_scaling_factor = exploration_scaling_factor

    
    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        
        if evaluate is False:
            action, _, _ = self.actor.sample(state)
        else:
            _, _, action = self.actor.sample(state)

        return action.detach().cpu().numpy()[0]


    def update_parameters(self, memory:ReplayBuffer, batch_size, updates):
        state_batch, next_state_batch, action_batch, reward_batch, terminated_batch, truncated_batch = memory.sample_buffer(batch_size)

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        terminated_batch = torch.FloatTensor(terminated_batch).to(self.device).unsqueeze(1)
        truncated_batch = torch.FloatTensor(truncated_batch).to(self.device).unsqueeze(1)

        predicted_next_state = self.icm(state_batch, action_batch)
        prediction_error = F.mse_loss(predicted_next_state, next_state_batch)
        prediction_error_no_reduction = F.mse_loss(predicted_next_state, next_state_batch, reduction='none')

        scaled_intrinsic_reward = prediction_error_no_reduction.mean(dim=1)
        scaled_intrinsic_reward = self.exploration_scaling_factor * torch.reshape(scaled_intrinsic_reward, (batch_size, 1))

        reward_batch += scaled_intrinsic_reward

        with torch.no_grad():
            next_action, next_log_prob, _ = self.actor.sample(next_state_batch)
            target_q1, target_q2 = self.critic_target(next_state_batch, next_action)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_prob
            target_q = reward_batch + (1 - terminated_batch.float()) * self.gamma * target_q

        q1, q2 = self.critic(state_batch, action_batch)
        q1_loss = F.mse_loss(q1, target_q)
        q2_loss = F.mse_loss(q2, target_q)
        q_loss = q1_loss + q2_loss

        self.critic_optim.zero_grad()
        q_loss.backward()
        self.critic_optim.step()

        self.icm_optim.zero_grad()
        prediction_error.backward()
        self.icm_optim.step()

        pi, log_pi, _ = self.actor.sample(state_batch)
        q1_pi, q2_pi = self.critic(state_batch, pi)
        min_q_pi = torch.min(q1_pi, q2_pi)

        actor_loss = (self.alpha * log_pi - min_q_pi).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        alpha_loss = torch.tensor(0.).to(self.device) 
        alpha_tlogs = torch.tensor(self.alpha) 

        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        return q1_loss.item(), q2_loss.item(), actor_loss.item(), alpha_loss.item(), prediction_error.item(), alpha_tlogs.item()      
    

    def train(self, env, memory: ReplayBuffer, episodes, batch_size, updates_per_step, summary_writer_name, max_episode_steps):
        warm_up = 20

        summary_writer_name = f'runs/{datetime.now().strftime("%Y%m%d-%H%M%S")}-{summary_writer_name}'
        writer = SummaryWriter(summary_writer_name)

        total_numsteps = 0
        updates = 0

        for episode in range(episodes):
            episode_reward = 0
            episode_steps = 0
            terminated = False
            truncated = False

            state, _ = env.reset()

            while not terminated and episode_steps < max_episode_steps:
                if warm_up > episode:
                    action = env.action_space.sample()
                else:
                    action = self.select_action(state)

                if memory.can_smaple(batch_size=batch_size):
                    for _ in range(updates_per_step):
                        critic1_loss, critic2_loss, actor_loss, ent_loss, prediction_loss, alpha = self.update_parameters(memory, batch_size, updates)
                        writer.add_scalar('Loss/Critic1', critic1_loss, updates)
                        writer.add_scalar('Loss/Critic2', critic2_loss, updates)
                        writer.add_scalar('Loss/Actor', actor_loss, updates)
                        writer.add_scalar('Loss/Entropy', ent_loss, updates)
                        writer.add_scalar('Loss/Prediction', prediction_loss, updates)
                        writer.add_scalar('Parameters/Alpha', alpha, updates)
                        updates += 1
                
                next_state, reward, terminated, truncated, _ = env.step(action)

                episode_steps += 1
                total_numsteps += 1
                episode_reward += reward

                memory.store_transition(state, next_state, action, reward, terminated, truncated)

                state = next_state

            writer.add_scalar('Reward/Train', episode_reward, episode)
            print(f'Episode {episode}, total numsteps: {total_numsteps}, episode steps: {episode_steps}, episode reward: {round(episode_reward, 2)}')

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

                # if reward == 1:
                #     terminated = True

                episode_reward += reward

                state = next_state

            print(f'Test Episode {episode}, steps: {episode_steps}, reward: {round(episode_reward, 2)}')

    def save_checkpoint(self):
        if not os.path.exists('checkpoints/'):
            os.makedirs('checkpoints/')
        
        print('Saving models')

        self.actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.critic_target.save_checkpoint()


    def load_checkpoint(self, evaluate=False):
        try:
            print("Loading checkpoints...")
            self.actor.load_checkpoint()
            self.critic.load_checkpoint()
            self.critic_target.load_checkpoint()
            print("Checkpoints loaded successfully.")
        except:
            if evaluate:
                raise Exception("No checkpoints found for evaluation.")
            else:
                print("No checkpoints found, starting from scratch.")
        
        if evaluate:
            self.actor.eval()
            self.critic.eval()
            self.critic_target.eval()
        else:
            self.actor.train()
            self.critic.train()
            self.critic_target.train()
