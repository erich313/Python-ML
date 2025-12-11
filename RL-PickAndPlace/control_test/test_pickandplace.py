import gymnasium as gym
import gymnasium_robotics

from test_wrapper import DataWrapper


if __name__ == "__main__":
    
    gym.register_envs(gymnasium_robotics)
    
    env = gym.make('FetchPickAndPlaceDense-v4', max_episode_steps=100, render_mode='human')
    print(env.unwrapped.dt)
    print(env.unwrapped.model.opt.timestep)
    env = DataWrapper(env)
    print(env.unwrapped.dt)
    print(env.unwrapped.model.opt.timestep)

    obs, info = env.reset()
    for _ in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        if terminated or truncated:
            obs, info = env.reset()
    

    env.close()
    