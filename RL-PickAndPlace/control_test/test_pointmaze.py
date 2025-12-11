import gymnasium as gym
import gymnasium_robotics

from test_wrapper import DataWrapper


if __name__ == "__main__":
    
    gym.register_envs(gymnasium_robotics)
    
    env = gym.make('PointMaze_UMaze-v3', max_episode_steps=50, render_mode='human')
    print(env.unwrapped.model.opt.timestep)
    env = DataWrapper(env)
    print(env.unwrapped.model.opt.timestep)

    obs, info = env.reset()
    for _ in range(200):
        action = [1, 0]
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        if terminated or truncated:
            obs, info = env.reset()
    

    env.close()
    