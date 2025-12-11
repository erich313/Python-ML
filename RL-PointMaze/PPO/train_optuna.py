import gymnasium as gym
import gymnasium_robotics
import numpy as np
import optuna

from wrapper import DataWrapper
from net import *
from agent_optuna import Agent
from buffer import *


def objective(trial: optuna.Trial) -> float:
    '''
    In optuna, conventionally functions to be optimized are named objective.
    Args:
        trial: A Trial object corresponds to a single execution of the objective function and is internally instantiated upon each invocation of the function.
            The suggest APIs (for example, suggest_float()) are called inside the objective function to obtain parameters for a trial.
    Returns:
        The final average score after training
    '''
    
    ## Suggest Hyperparameters 
    gamma = trial.suggest_float("gamma", 0.98, 0.999, log=True)
    lambdaa = trial.suggest_float("lambdaa", 0.9, 0.99)
    epsilon = trial.suggest_float("epsilon", 0.1, 0.3)
    entropy_coef = trial.suggest_float("entropy_coef", 1e-4, 1e-1, log=True)
    epochs = trial.suggest_int("epochs", 3, 10)
    hidden_dim = trial.suggest_categorical("hidden_dim", [256, 512])
    learning_rate_ac = trial.suggest_float("learning_rate_ac", 1e-5, 1e-3, log=True)
    learning_rate_cr = trial.suggest_float("learning_rate_cr", 1e-5, 1e-3, log=True)
    
    batch_size = trial.suggest_categorical("batch_size", [1024, 2048, 4096])
    mini_batch_size = trial.suggest_categorical("mini_batch_size", [128, 256, 512])
    
    # Ensure mini_batch_size is valid and Enforce that mini_batch_size divides batch_size.
    if mini_batch_size > batch_size or batch_size % mini_batch_size != 0:
        raise optuna.exceptions.TrialPruned()

    ## Fixed parameters
    max_episode_steps = 100
    episodes = 5000

    STRAIGHT_MAZE = [[1, 1, 1, 1, 1],
                     [1, 0, 0, 0, 1],
                     [1, 1, 1, 1, 1]]
    
    gym.register_envs(gymnasium_robotics)
    env = gym.make('PointMaze_UMaze-v3', max_episode_steps=max_episode_steps, maze_map=STRAIGHT_MAZE)
    env = DataWrapper(env)

    observation, info = env.reset()

    observation_dim = observation.shape[0]
    action_dim=env.action_space.shape[0]

    agent = Agent(state_dim=observation_dim, 
                  action_space=env.action_space,
                  hidden_dim=hidden_dim,            
                  learning_rate_ac=learning_rate_ac, 
                  learning_rate_cr=learning_rate_cr,
                  gamma=gamma,                      
                  lambdaa=lambdaa,                 
                  epsilon=epsilon,                   
                  entropy_coef=entropy_coef,         
                  epochs=epochs)            
    
    buffer = RolloutBuffer(batch_size=batch_size, 
                           mini_batch_size=mini_batch_size, 
                           state_dim=observation_dim,
                           action_dim=action_dim)

    ## Start the optimization to maximize cumulative reward
    try:
        score = agent.train(env,
                                  episodes=episodes,
                                  max_episode_steps=max_episode_steps,
                                  buffer=buffer,
                                  trial=trial)
    except optuna.exceptions.TrialPruned:
        raise
    except Exception as e:
        return -np.inf
    finally:
        env.close()

    return score


if __name__ == "__main__":
    # In Optuna, we use the study object to manage optimization. Method create_study() returns a study object.
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=10)
    study = optuna.create_study(
        direction="maximize",
        pruner=pruner,
        storage="sqlite:///studies/ppo_study.db",
        study_name="ppo-pointmaze-test", 
        load_if_exists=True 
    )
    
    study.optimize(objective, n_trials=100) 

    print("\nStudy statistics: ")
    print(f"  Number of finished trials: {len(study.trials)}")
    
    print("\nBest trial:")
    best_trial = study.best_trial
    
    print(f"  Value (Max Avg Reward): {best_trial.value}")
    
    print("\nBest Hyperparameters: ")
    for key, value in best_trial.params.items():
        print(f"  {key}: {value}")
