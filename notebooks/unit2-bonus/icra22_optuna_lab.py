# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown] id="hyyN-2qyK_T2"
# # Hyperparameter tuning with Optuna
#
# Github repo: https://github.com/araffin/tools-for-robotic-rl-icra2022
#
# Optuna: https://github.com/optuna/optuna
#
# Stable-Baselines3: https://github.com/DLR-RM/stable-baselines3
#
# Documentation: https://stable-baselines3.readthedocs.io/en/master/
#
# SB3 Contrib: https://github.com/Stable-Baselines-Team/stable-baselines3-contrib
#
# RL Baselines3 zoo: https://github.com/DLR-RM/rl-baselines3-zoo
#
# [RL Baselines3 Zoo](https://github.com/DLR-RM/rl-baselines3-zoo) is a collection of pre-trained Reinforcement Learning agents using Stable-Baselines3.
#
# It also provides basic scripts for training, evaluating agents, tuning hyperparameters and recording videos.
#
#
# ## Introduction
#
# In this notebook, you will learn the importance of tuning hyperparameters. You will first try to optimize the parameters manually and then we will see how to automate the search using Optuna.
#
#
# ## Install Dependencies and Stable Baselines3 Using Pip
#
# List of full dependencies can be found in the [README](https://github.com/DLR-RM/stable-baselines3).
#
#
# ```
# pip install stable-baselines3[extra]
# ```

# %% id="hYdv2ygjLaFL" vscode={"languageId": "python"}
# !pip install stable-baselines3

# %% id="oexj67yWN5_k" vscode={"languageId": "python"}
# Optional: install SB3 contrib to have access to additional algorithms
# !pip install sb3-contrib

# %% id="NNah91r9x9EL" vscode={"languageId": "python"}
# Optuna will be used in the last part when doing hyperparameter tuning
# !pip install optuna

# %% [markdown] id="FtY8FhliLsGm"
# ## Imports

# %% id="BIedd7Pz9sOs" vscode={"languageId": "python"}
import gym
import numpy as np

# %% [markdown] id="Ae32CtgzTG3R"
# The first thing you need to import is the RL model, check the documentation to know what you can use on which problem

# %% id="R7tKaBFrTR0a" vscode={"languageId": "python"}
from stable_baselines3 import PPO, A2C, SAC, TD3, DQN

# %% id="EcsXmYRMON9W" vscode={"languageId": "python"}
# Algorithms from the contrib repo
# https://github.com/Stable-Baselines-Team/stable-baselines3-contrib
from sb3_contrib import QRDQN, TQC

# %% id="kLwjcfvuqtGE" vscode={"languageId": "python"}
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

# %% [markdown] id="-khNkrgcI6Z1"
# # Part I: The Importance Of Tuned Hyperparameters
#
#

# %% [markdown] id="PytOtL9GdmrE"
# When compared with Supervised Learning, Deep Reinforcement Learning is far more sensitive to the choice of hyper-parameters such as learning rate, number of neurons, number of layers, optimizer ... etc.
#
# Poor choice of hyper-parameters can lead to poor/unstable convergence. This challenge is compounded by the variability in performance across random seeds (used to initialize the network weights and the environment).

# %% [markdown] id="Hk8HSIC3qUjc"
# In addition to hyperparameters, selecting the appropriate algorithm is also an important choice. We will demonstrate it on the simple Pendulum task.
#
# See [gym doc](https://gym.openai.com/envs/Pendulum-v0/): "The inverted pendulum swingup problem is a classic problem in the control literature. In this version  of the problem, the pendulum starts in a random position, and the goal is to swing it up so it stays upright."
#
#
# Let's try first with PPO and a small budget of 4000 steps (20 episodes):

# %% id="4ToIvihGq2N0" vscode={"languageId": "python"}
env_id = "Pendulum-v1"
# Env used only for evaluation
eval_envs = make_vec_env(env_id, n_envs=10)
# 4000 training timesteps
budget_pendulum = 4000

# %% [markdown] id="EWT2r6QE4yew"
# ### PPO

# %% id="KCHk_-_4ndux" vscode={"languageId": "python"}
ppo_model = PPO("MlpPolicy", env_id, seed=0, verbose=0).learn(budget_pendulum)

# %% colab={"base_uri": "https://localhost:8080/"} id="TP9C9AqLndxz" outputId="dd8e423c-dd4d-43cf-eac5-639e6748f02c" vscode={"languageId": "python"}
mean_reward, std_reward = evaluate_policy(ppo_model, eval_envs, n_eval_episodes=100, deterministic=True)

print(f"PPO Mean episode reward: {mean_reward:.2f} +/- {std_reward:.2f}")

# %% [markdown] id="uHmJaJLl5ds4"
# ### A2C

# %% id="BLL_pws25jh0" vscode={"languageId": "python"}
# Define and train a A2C model
a2c_model = A2C("MlpPolicy", env_id, seed=0, verbose=0).learn(budget_pendulum)

# %% id="ic83jZwB5nVk" vscode={"languageId": "python"}
# Evaluate the train A2C model
mean_reward, std_reward = evaluate_policy(a2c_model, eval_envs, n_eval_episodes=100, deterministic=True)

print(f"A2C Mean episode reward: {mean_reward:.2f} +/- {std_reward:.2f}")

# %% [markdown] id="0_z1zFx2rVpG"
# Both are far from solving the env (mean reward around -200).
# Now, let's try with an off-policy algorithm:

# %% [markdown] id="3wYaVZJU5VL5"
# ### Training longer PPO ?
#
# Maybe training longer would help?
#
# You can try with 10x the budget, but in the case of A2C/PPO, training longer won't help much, finding better hyperparameters is needed instead.

# %% id="hHsHpnQY6TWA" vscode={"languageId": "python"}
# train longer
new_budget = 10 * budget_pendulum

ppo_model = PPO("MlpPolicy", env_id, seed=0, verbose=0).learn(new_budget)

# %% id="7OD9y1o36Xta" vscode={"languageId": "python"}
mean_reward, std_reward = evaluate_policy(ppo_model, eval_envs, n_eval_episodes=100, deterministic=True)

print(f"PPO Mean episode reward: {mean_reward:.2f} +/- {std_reward:.2f}")

# %% [markdown] id="YEvQ9SJ15Xmh"
# ### PPO - Tuned Hyperparameters
#
# Using Optuna, we can in fact tune the hyperparameters and find a working solution (from the [RL Zoo](https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/ppo.yml)):

# %% id="S-D_vvsb6jOZ" vscode={"languageId": "python"}
tuned_params = {
    "gamma": 0.9,
    "use_sde": True,
    "sde_sample_freq": 4,
    "learning_rate": 1e-3,
}

# budget = 10 * budget_pendulum
ppo_tuned_model = PPO("MlpPolicy", env_id, seed=1, verbose=1, **tuned_params).learn(50_000, log_interval=5)

# %% colab={"base_uri": "https://localhost:8080/"} id="MLuxoLxt67xO" outputId="6bc7479b-689f-4d0f-9f01-379c31afdb4e" vscode={"languageId": "python"}
mean_reward, std_reward = evaluate_policy(ppo_tuned_model, eval_envs, n_eval_episodes=100, deterministic=True)

print(f"Tuned PPO Mean episode reward: {mean_reward:.2f} +/- {std_reward:.2f}")

# %% [markdown] id="2H33u_apWPp5"
# Note: if you try SAC on the simple MountainCarContinuous environment, you will encounter some issues without tuned hyperparameters: https://github.com/rail-berkeley/softlearning/issues/76
#
# Simple environments can be challenging even for SOTA algorithms.

# %% [markdown] id="_vdpPJ04nebx"
# # Part II: Grad Student Descent
#

# %% [markdown] id="n8PNN9kcgolk"
# ### Challenge (10 minutes): "Grad Student Descent"
# The challenge is to find the best hyperparameters (max performance) for A2C on `CartPole-v1` with a limited budget of 20 000 training steps.
#
#
# Maximum reward: 500 on `CartPole-v1`
#
# The hyperparameters should work for different random seeds.

# %% id="s6aqxsini7H3" vscode={"languageId": "python"}
budget = 20_000

# %% [markdown] id="yDQ805DBi3KM"
# #### The baseline: default hyperparameters

# %% id="pyOCKf4Vt-HK" vscode={"languageId": "python"}
eval_envs_cartpole = make_vec_env("CartPole-v1", n_envs=10)

# %% id="D1PSNGcsi2dP" vscode={"languageId": "python"}
# %%time
model = A2C("MlpPolicy", "CartPole-v1", seed=42, verbose=1).learn(budget)

# %% colab={"base_uri": "https://localhost:8080/"} id="2d3X0G0ng2OE" outputId="8d550b14-a673-4abd-b9b8-c539d9c79c05" vscode={"languageId": "python"}
mean_reward, std_reward = evaluate_policy(model, eval_envs_cartpole, n_eval_episodes=50, deterministic=True)

print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")

# %% [markdown] id="B-fi1-oKnUI2"
# **Your goal is to beat that baseline and get closer to the optimal score of 500**

# %% [markdown] id="qvq8zizok1X_"
# Time to tune!

# %% id="UaqCCH4gkRH_" vscode={"languageId": "python"}
import torch.nn as nn

# %% id="uDUfeZcyjPKS" vscode={"languageId": "python"}

policy_kwargs = dict(
    net_arch=[
      dict(vf=[64, 64], pi=[64, 64]), # network architectures for actor/critic
    ],
    activation_fn=nn.Tanh,
)

hyperparams = dict(
    n_steps=5, # number of steps to collect data before updating policy
    learning_rate=7e-4,
    gamma=0.99, # discount factor
    max_grad_norm=0.5, # The maximum value for the gradient clipping
    ent_coef=0.0, # Entropy coefficient for the loss calculation
)

model = A2C("MlpPolicy", "CartPole-v1", seed=7, verbose=1, **hyperparams).learn(budget, progress_bar=True)

# %% vscode={"languageId": "python"} id="6uKbrjNCrvHK"
mean_reward, std_reward = evaluate_policy(model, eval_envs_cartpole, n_eval_episodes=50, deterministic=True)

print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")

# %% [markdown] id="iL_G9DurUV75"
# Hint - Recommended Hyperparameter Range
#
# ```python
# gamma = trial.suggest_float("gamma", 0.9, 0.99999, log=True)
# max_grad_norm = trial.suggest_float("max_grad_norm", 0.3, 5.0, log=True)
# # from 2**3 = 8 to 2**10 = 1024
# n_steps = 2 ** trial.suggest_int("exponent_n_steps", 3, 10)
# learning_rate = trial.suggest_float("lr", 1e-5, 1, log=True)
# ent_coef = trial.suggest_float("ent_coef", 0.00000001, 0.1, log=True)
# # net_arch tiny: {"pi": [64], "vf": [64]}
# # net_arch default: {"pi": [64, 64], "vf": [64, 64]}
# # activation_fn = nn.Tanh / nn.ReLU
# ```

# %% [markdown] id="QwFOp0j-ga-_"
# # Part III: Automatic Hyperparameter Tuning
#
#
#
#

# %% [markdown] id="88x7wMyyud5p"
# In this part we will create a script that allows to search for the best hyperparameters automatically.

# %% [markdown] id="auwR-30IvHeY"
# ### Imports

# %% id="VM6tUr-yuekR" vscode={"languageId": "python"}
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from optuna.visualization import plot_optimization_history, plot_param_importances

# %% [markdown] id="ZQVfmM1dzA1d"
# ### Config

# %% id="yyBTVcAGzCRk" vscode={"languageId": "python"}
N_TRIALS = 100  # Maximum number of trials
N_JOBS = 1 # Number of jobs to run in parallel
N_STARTUP_TRIALS = 5  # Stop random sampling after N_STARTUP_TRIALS
N_EVALUATIONS = 2  # Number of evaluations during the training
N_TIMESTEPS = int(2e4)  # Training budget
EVAL_FREQ = int(N_TIMESTEPS / N_EVALUATIONS)
N_EVAL_ENVS = 5
N_EVAL_EPISODES = 10
TIMEOUT = int(60 * 15)  # 15 minutes

ENV_ID = "CartPole-v1"

DEFAULT_HYPERPARAMS = {
    "policy": "MlpPolicy",
    "env": ENV_ID,
}

# %% [markdown] id="25HgcDYzvJ0b"
# ### Exercise (5 minutes): Define the search space

# %% id="KXo8AwGAvN8Q" vscode={"languageId": "python"}
from typing import Any, Dict
import torch
import torch.nn as nn

def sample_a2c_params(trial: optuna.Trial) -> Dict[str, Any]:
    """
    Sampler for A2C hyperparameters.

    :param trial: Optuna trial object
    :return: The sampled hyperparameters for the given trial.
    """
    # Discount factor between 0.9 and 0.9999
    gamma = 1.0 - trial.suggest_float("gamma", 0.0001, 0.1, log=True)
    max_grad_norm = trial.suggest_float("max_grad_norm", 0.3, 5.0, log=True)
    # 8, 16, 32, ... 1024
    n_steps = 2 ** trial.suggest_int("exponent_n_steps", 3, 10)

    ### YOUR CODE HERE
    # TODO:
    # - define the learning rate search space [1e-5, 1] (log) -> `suggest_float`
    # - define the network architecture search space ["tiny", "small"] -> `suggest_categorical`
    # - define the activation function search space ["tanh", "relu"]
    learning_rate = trial.suggest_float("lr", 1e-5, 1, log=True)
    net_arch = trial.suggest_categorical("net_arch", ["tiny", "small"])
    activation_fn = trial.suggest_categorical("act_fn", ["tanh", "relu"])

    ### END OF YOUR CODE

    # Display true values
    trial.set_user_attr("gamma_", gamma)
    trial.set_user_attr("n_steps", n_steps)

    net_arch = [
        {"pi": [64], "vf": [64]}
        if net_arch == "tiny"
        else {"pi": [64, 64], "vf": [64, 64]}
    ]

    activation_fn = {"tanh": nn.Tanh, "relu": nn.ReLU}[activation_fn]

    return {
        "n_steps": n_steps,
        "gamma": gamma,
        "learning_rate": learning_rate,
        "max_grad_norm": max_grad_norm,
        "policy_kwargs": {
            "net_arch": net_arch,
            "activation_fn": activation_fn,
        },
    }


# %% [markdown] id="iybymNiJxNu7"
# ### Define the objective function

# %% [markdown] id="YJY8Z8tuxai7"
# First we define a custom callback to report the results of periodic evaluations to Optuna:

# %% id="U5ijWTPzxSmd" vscode={"languageId": "python"}
from stable_baselines3.common.callbacks import EvalCallback

class TrialEvalCallback(EvalCallback):
    """
    Callback used for evaluating and reporting a trial.

    :param eval_env: Evaluation environement
    :param trial: Optuna trial object
    :param n_eval_episodes: Number of evaluation episodes
    :param eval_freq:   Evaluate the agent every ``eval_freq`` call of the callback.
    :param deterministic: Whether the evaluation should
        use a stochastic or deterministic policy.
    :param verbose:
    """

    def __init__(
        self,
        eval_env: gym.Env,
        trial: optuna.Trial,
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        deterministic: bool = True,
        verbose: int = 0,
    ):

        super().__init__(
            eval_env=eval_env,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            deterministic=deterministic,
            verbose=verbose,
        )
        self.trial = trial
        self.eval_idx = 0
        self.is_pruned = False

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Evaluate policy (done in the parent class)
            super()._on_step()
            self.eval_idx += 1
            # Send report to Optuna
            self.trial.report(self.last_mean_reward, self.eval_idx)
            # Prune trial if need
            if self.trial.should_prune():
                self.is_pruned = True
                return False
        return True


# %% [markdown] id="8cHNM_cFO3vs"
# ### Exercise (10 minutes): Define the objective function

# %% [markdown] id="76voi9AXxlCq"
# Then we define the objective function that is in charge of sampling hyperparameters, creating the model and then returning the result to Optuna

# %% id="E0yEokTDxhrC" vscode={"languageId": "python"}
def objective(trial: optuna.Trial) -> float:
    """
    Objective function using by Optuna to evaluate
    one configuration (i.e., one set of hyperparameters).

    Given a trial object, it will sample hyperparameters,
    evaluate it and report the result (mean episodic reward after training)

    :param trial: Optuna trial object
    :return: Mean episodic reward after training
    """

    kwargs = DEFAULT_HYPERPARAMS.copy()
    
    ### YOUR CODE HERE
    # TODO:
    # 1. Sample hyperparameters and update the default keyword arguments: `kwargs.update(other_params)`
    # 2. Create the evaluation envs
    # 3. Create the `TrialEvalCallback`

    # 1. Sample hyperparameters and update the keyword arguments
    kwargs.update(sample_a2c_params(trial))
    
    # Create the RL model
    model = A2C(**kwargs)

    # 2. Create envs used for evaluation using `make_vec_env`, `ENV_ID` and `N_EVAL_ENVS`
    eval_env = make_vec_env("CartPole-v1", n_envs=10)
    
    # 3. Create the `TrialEvalCallback` callback defined above that will periodically evaluate
    # and report the performance using `N_EVAL_EPISODES` every `EVAL_FREQ`
    # TrialEvalCallback signature:
    # TrialEvalCallback(eval_env, trial, n_eval_episodes, eval_freq, deterministic, verbose)
    
    trial_eval_cb = TrialEvalCallback(eval_env, trial)
    eval_callback = trial_eval_cb

    ### END OF YOUR CODE

    nan_encountered = False
    try:
        # Train the model
        model.learn(N_TIMESTEPS, callback=eval_callback, progress_bar=True)
    except AssertionError as e:
        # Sometimes, random hyperparams can generate NaN
        print(e)
        nan_encountered = True
    finally:
        # Free memory
        model.env.close()
        eval_envs.close()

    # Tell the optimizer that the trial failed
    if nan_encountered:
        return float("nan")

    if eval_callback.is_pruned:
        raise optuna.exceptions.TrialPruned()

    return eval_callback.last_mean_reward


# %% [markdown] id="jMFLu_M0ymzj"
# ### The optimization loop

# %% id="4UU17YpjymPr" vscode={"languageId": "python"}
# %%time

import torch as th

# Set pytorch num threads to 1 for faster training
th.set_num_threads(1)
# Select the sampler, can be random, TPESampler, CMAES, ...
sampler = TPESampler(n_startup_trials=N_STARTUP_TRIALS)
# Do not prune before 1/3 of the max budget is used
pruner = MedianPruner(
    n_startup_trials=N_STARTUP_TRIALS, n_warmup_steps=N_EVALUATIONS // 3
)
# Create the study and start the hyperparameter optimization
study = optuna.create_study(sampler=sampler, pruner=pruner, direction="maximize")

try:
    study.optimize(objective, n_trials=N_TRIALS, n_jobs=N_JOBS, timeout=TIMEOUT)
except KeyboardInterrupt:
    pass

# %% id="4UU17YpjymPr" vscode={"languageId": "python"}
print("Number of finished trials: ", len(study.trials))

print("Best trial:")
trial = study.best_trial

print(f"  Value: {trial.value}")

print("  Params: ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")

print("  User attrs:")
for key, value in trial.user_attrs.items():
    print(f"    {key}: {value}")

# Write report
study.trials_dataframe().to_csv("study_results_a2c_cartpole.csv")

fig1 = plot_optimization_history(study)
fig2 = plot_param_importances(study)

fig1.show()
fig2.show()

# %%
plot_optimization_history(study)

# %% [markdown] id="SCbep6z1h3D1"
# Complete example: https://github.com/DLR-RM/rl-baselines3-zoo

# %% [markdown] id="7yUeYnfJVpB2"
# # Conclusion
#
# What we have seen in this notebook:
# - the importance of good hyperparameters
# - how to do automatic hyperparameter search with optuna
#

# %% id="3-gqIPXqV7zZ" vscode={"languageId": "python"}
