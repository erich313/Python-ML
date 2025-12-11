import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


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
        plt.savefig('frozen_lake_td.png')
        plt.close()
    
    def graphing_visited(self, visited):
        plt.figure(2)
        v = np.zeros((8, 8))
        for i in range(8):
            for j in range(8):
                vv = visited[i*8+j]
                v[i][j] = np.average(vv[vv!=0])
        sns.heatmap(v, annot=True, fmt=".0f", cmap="Blues", cbar=1)
        plt.title('Average action taken in each state')
        plt.savefig('visited.png')
        plt.close()

    def graphing_monitor(self, monitor_state, monitor_state_number):
        plt.figure(3)

        colors = ['red', 'green', 'blue', 'orange']
        for i, arr in enumerate(monitor_state):
            plt.plot(arr, color=colors[i], label=f'action {i}')

        plt.legend()
        plt.title(f"Monitoring Q-values for state {monitor_state_number}")
        plt.xlabel("Episode")
        plt.ylabel("Q-value")
        plt.savefig(f'state{monitor_state_number}.png')

        plt.close()
