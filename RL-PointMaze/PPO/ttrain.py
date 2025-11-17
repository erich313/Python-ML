import gymnasium as gym
import gymnasium_robotics
import numpy as np
import optuna # <-- Import Optuna

from wrapper import DataWrapper
from net import *
from tagent import Agent
from buffer import *

# --- 1. Define the Objective Function ---
# We move ALL training logic inside this function
def objective(trial: optuna.Trial) -> float:
    
    # --- A. Suggest Hyperparameters ---
    # Optuna will pick values for these
    gamma = trial.suggest_float("gamma", 0.98, 0.999, log=True)
    lambdaa = trial.suggest_float("lambdaa", 0.9, 0.99)
    epsilon = trial.suggest_float("epsilon", 0.1, 0.3)
    entropy_coef = trial.suggest_float("entropy_coef", 1e-4, 1e-1, log=True)
    epochs = trial.suggest_int("epochs", 3, 10)
    hidden_dim = trial.suggest_categorical("hidden_dim", [256, 512])
    learning_rate_ac = trial.suggest_float("learning_rate_ac", 1e-5, 1e-3, log=True)
    learning_rate_cr = trial.suggest_float("learning_rate_cr", 1e-5, 1e-3, log=True)
    
    # Dependent hyperparameters: mini_batch_size must be a divisor of batch_size
    batch_size = trial.suggest_categorical("batch_size", [1024, 2048, 4096])
    mini_batch_size = trial.suggest_categorical("mini_batch_size", [128, 256, 512])
    
    # Ensure mini_batch_size is valid
    if mini_batch_size > batch_size:
        # If invalid combo, prune this trial early
        raise optuna.exceptions.TrialPruned()
    if batch_size % mini_batch_size != 0:
        # If not a clean divisor, prune
        raise optuna.exceptions.TrialPruned()

    # --- B. Set up Environment & Agent (using suggested params) ---
    max_episode_steps = 100 # Keep this fixed for fair comparison
    episodes = 4000         # Fixed number of episodes per trial

    STRAIGHT_MAZE = [[1, 1, 1, 1, 1],
                     [1, 0, 0, 0, 1],
                     [1, 1, 1, 1, 1]]
    
    gym.register_envs(gymnasium_robotics)
    # Set render_mode='rgb_array' or remove it to speed up training
    env = gym.make('PointMaze_UMaze-v3', max_episode_steps=max_episode_steps, maze_map=STRAIGHT_MAZE)
    env = DataWrapper(env)

    observation, info = env.reset()
    observation_dim = observation.shape[0]
    action_dim=env.action_space.shape[0]

    agent = Agent(state_dim=observation_dim, 
                  action_space=env.action_space,
                  hidden_dim=hidden_dim,             # <-- Use trial param
                  learning_rate_ac=learning_rate_ac, # <-- Use trial param
                  learning_rate_cr=learning_rate_cr, # <-- Use trial param
                  gamma=gamma,                       # <-- Use trial param
                  lambdaa=lambdaa,                   # <-- Use trial param
                  epsilon=epsilon,                   # <-- Use trial param
                  entropy_coef=entropy_coef,         # <-- Use trial param
                  epochs=epochs)                     # <-- Use trial param
    
    buffer = RolloutBuffer(batch_size=batch_size,         # <-- Use trial param
                           mini_batch_size=mini_batch_size, # <-- Use trial param
                           state_dim=observation_dim,
                           action_dim=action_dim)

    # --- C. Run Training and Get Score ---
    try:
        # Pass the 'trial' object to your modified train function
        final_score = agent.train(env,
                                  episodes=episodes,
                                  summary_writer_name=f'trial_{trial.number}',
                                  max_episode_steps=max_episode_steps,
                                  buffer=buffer,
                                  trial=trial) # <-- Pass trial for pruning
    except optuna.exceptions.TrialPruned:
        # Catch the prune exception and re-raise it
        raise
    except Exception as e:
        # Handle other training errors
        print(f"Trial {trial.number} failed with error: {e}")
        return -np.inf # Return a very bad score
    finally:
        env.close()

    # --- D. Return the Final Score ---
    return final_score


# --- 2. Create and Run the Optuna Study ---
if __name__ == "__main__":
    # Create a study. We want to MAXIMIZE the average reward.
    # We also add a Pruner to stop bad trials early
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=10) # Prune after 10 reports (10*100=1000 episodes)
    study = optuna.create_study(
        direction="maximize",
        pruner=pruner,
        storage="sqlite:///ppo_study.db",  # <-- 儲存到一個名為 'ppo_study.db' 的檔案
        study_name="ppo-pointmaze-v2",      # <-- 給這個 study 一個名稱
        load_if_exists=True                   # <-- 如果檔案存在，就繼續上次的 study
    )
    
    # Start the optimization
    # n_trials is the total number of hyperparameter sets to test
    study.optimize(objective, n_trials=100) 

    # --- 3. Print the Best Results ---
    print("\nStudy statistics: ")
    print(f"  Number of finished trials: {len(study.trials)}")
    
    print("\nBest trial:")
    best_trial = study.best_trial
    
    print(f"  Value (Max Avg Reward): {best_trial.value}")
    
    print("  Best Hyperparameters: ")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")

# v1 2500 episode
# v2 4000 episode