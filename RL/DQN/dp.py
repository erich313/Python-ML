import numpy as np
import matplotlib.pyplot as plt


class DataProcessor:
    def __init__(self, episodes):
        self.rewards = np.zeros(episodes)
        self.epsilons = []
        self.episodes = episodes

    def track_rewards(self, episode, reward):
        self.rewards[episode] = reward

    def sum_rewards(self):
        return np.sum(self.rewards)

    def track_epsilon(self, epsilon):
        self.epsilons.append(epsilon)

    def graphing(self):
        # Create new graph 
        plt.figure(1)

        # Plot average rewards (Y-axis) vs episodes (X-axis)
        sum_rewards = np.zeros(self.episodes)
        for x in range(self.episodes):
            sum_rewards[x] = np.sum(self.rewards[max(0, x-100):(x+1)])
        plt.subplot(121) # plot on a 1 row x 2 col grid, at cell 1
        plt.plot(sum_rewards)
        
        # Plot epsilon decay (Y-axis) vs episodes (X-axis)
        plt.subplot(122) # plot on a 1 row x 2 col grid, at cell 2
        plt.plot(self.epsilons)
        
        # Save plots
        plt.savefig('frozen_lake_dql.png')