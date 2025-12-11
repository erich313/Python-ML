from gymnasium import Wrapper
import numpy as np


class CustomEnvWrapper(Wrapper):
    def __init__(self, env, slippery):
        super().__init__(env)
        self.perpandicular = {
            0: [1, 3],  # left -> down, up
            1: [0, 2],  # down -> left, right
            2: [1, 3],  # right -> down, up
            3: [0, 2]   # up -> left, right
        }
        self.slippery = slippery

    def step(self, action):    
        if self.slippery:   
            # Add some stochasticity
            rand = np.random.rand()
            if rand < 0.1:
                # 10% chance to go the first perpendicular way
                action = self.perpandicular[action][0]
            elif rand < 0.2:
                # 10% chance to go the second perpendicular way
                action = self.perpandicular[action][1]
            else:
                # 80% chance to go the original way
                pass
        else:
            # No stochasticity
            pass

        # Actually take a step in the environment
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Modify the reward function
        desc = self.env.unwrapped.desc
        nrow, ncol = self.env.unwrapped.nrow, self.env.unwrapped.ncol
        row, col = np.unravel_index(obs, (nrow, ncol))
        tile = desc[row][col].decode("utf-8")

        if tile == "H":
            reward = -10
        elif tile == "G":
            reward = 20
        else:
            reward = -0.1

        # Return obs(next_state), modified reward, terminated, truncated, and info
        return obs, reward, terminated, truncated, info
    