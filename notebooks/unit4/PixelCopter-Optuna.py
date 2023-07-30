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

# %% id="V8oadoJSWp7C"
import numpy as np

from collections import deque

import matplotlib.pyplot as plt

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

# Gym
import gym
import gym_pygame

# Hugging Face Hub
from huggingface_hub import notebook_login # To log to our Hugging Face account to be able to upload models to the Hub.
import imageio

from tqdm.auto import tqdm

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from optuna.visualization import plot_optimization_history, plot_param_importances

# %%
import sys
sys.path.append("../..")
from util import *

# %% [markdown] id="ZQVfmM1dzA1d"
# ### Config

# %% id="yyBTVcAGzCRk" vscode={"languageId": "python"} tags=["parameters"]
N_TRIALS = 100  # Maximum number of trials
N_JOBS = 1 # Number of jobs to run in parallel
N_STARTUP_TRIALS = 5  # Stop random sampling after N_STARTUP_TRIALS
N_EVALUATIONS = 10  # Number of evaluations during the training
N_TIMESTEPS = int(2e4)  # Training budget
EVAL_FREQ = int(N_TIMESTEPS / N_EVALUATIONS)
N_EVAL_ENVS = 5
N_EVAL_EPISODES = 10
TIMEOUT = int(60 * 60 * 6) # 4 hours

# %% [markdown] id="RfxJYdMeeVgv"
# ## Check if we have a GPU
#
# - Let's check if we have a GPU
# - If it's the case you should see `device:cuda0`

# %% id="kaJu5FeZxXGY"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# %% id="U5TNYa14aRav"
print(device)


# %% [markdown] id="PBPecCtBL_pZ"
# We're now ready to implement our Reinforce algorithm 🔥

# %% id="NCNvyElRStWG"
def reinforce(env, policy, optimizer, n_training_episodes, max_t, gamma, print_every, *, reset_info=False, callbacks=[]):
    # Help us to calculate the score during the training
    scores_deque = deque(maxlen=100)
    scores = []
    # Line 3 of pseudocode
    for i_episode in (pbar := tqdm(range(1, n_training_episodes+1))):
        saved_log_probs = []
        rewards = []
        if reset_info:
            state, *_ = env.reset()
        else:
            state = env.reset()
        # Line 4 of pseudocode
        for t in range(max_t):
            action, log_prob = policy.act(state)
            saved_log_probs.append(log_prob)
            state, reward, done, *_ = env.step(action)
                
            rewards.append(reward)
            if done:
                break 
        scores_deque.append(sum(rewards))
        scores.append(sum(rewards))
        
        # Line 6 of pseudocode: calculate the return
        returns = deque(maxlen=max_t) 
        n_steps = len(rewards) 
        # Compute the discounted returns at each timestep,
        # as 
        #      the sum of the gamma-discounted return at time t (G_t) + the reward at time t
        #
        # In O(N) time, where N is the number of time steps
        # (this definition of the discounted return G_t follows the definition of this quantity 
        # shown at page 44 of Sutton&Barto 2017 2nd draft)
        # G_t = r_(t+1) + r_(t+2) + ...
        
        # Given this formulation, the returns at each timestep t can be computed 
        # by re-using the computed future returns G_(t+1) to compute the current return G_t
        # G_t = r_(t+1) + gamma*G_(t+1)
        # G_(t-1) = r_t + gamma* G_t
        # (this follows a dynamic programming approach, with which we memorize solutions in order 
        # to avoid computing them multiple times)
        
        # This is correct since the above is equivalent to (see also page 46 of Sutton&Barto 2017 2nd draft)
        # G_(t-1) = r_t + gamma*r_(t+1) + gamma*gamma*r_(t+2) + ...
        
        
        ## Given the above, we calculate the returns at timestep t as: 
        #               gamma[t] * return[t] + reward[t]
        #
        ## We compute this starting from the last timestep to the first, in order
        ## to employ the formula presented above and avoid redundant computations that would be needed 
        ## if we were to do it from first to last.
        
        ## Hence, the queue "returns" will hold the returns in chronological order, from t=0 to t=n_steps
        ## thanks to the appendleft() function which allows to append to the position 0 in constant time O(1)
        ## a normal python list would instead require O(N) to do this.
        for t in range(n_steps)[::-1]:
            disc_return_t = (returns[0] if len(returns)>0 else 0)
            returns.appendleft( gamma*disc_return_t + rewards[t]   )    
            
        ## standardization of the returns is employed to make training more stable
        eps = np.finfo(np.float32).eps.item()
        ## eps is the smallest representable float, which is 
        # added to the standard deviation of the returns to avoid numerical instabilities        
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + eps)
        
        # Line 7:
        policy_loss = []
        for log_prob, disc_return in zip(saved_log_probs, returns):
            policy_loss.append(-log_prob * disc_return)
        policy_loss = torch.cat(policy_loss).sum()
        
        # Line 8: PyTorch prefers gradient descent 
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()
        
        for cb in callbacks:
            if not cb._on_step():
                return scores
        
        # if i_episode % print_every == 0:
        #     print('Episode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
        pbar.set_postfix({
            'train_score': np.mean(scores_deque),
        })
        
    return scores

# %% [markdown] id="JNLVmKKVKA6j"
# ## Second agent: PixelCopter 🚁
#
# ### Study the PixelCopter environment 👀
# - [The Environment documentation](https://pygame-learning-environment.readthedocs.io/en/latest/user/games/pixelcopter.html)
#

# %% id="JBSc8mlfyin3"
env_id = "Pixelcopter-PLE-v0"
env = gym.make(env_id)
eval_env = gym.make(env_id)
s_size = env.observation_space.shape[0]
a_size = env.action_space.n

# %% id="L5u_zAHsKBy7"
print("_____OBSERVATION SPACE_____ \n")
print("The State Space is: ", s_size)
print("Sample observation", env.observation_space.sample()) # Get a random observation

# %% id="D7yJM9YXKNbq"
print("\n _____ACTION SPACE_____ \n")
print("The Action Space is: ", a_size)
print("Action Space Sample", env.action_space.sample()) # Take a random action


# %% [markdown] id="NNWvlyvzalXr"
# The observation space (7) 👀:
# - player y position
# - player velocity
# - player distance to floor
# - player distance to ceiling
# - next block x distance to player
# - next blocks top y location
# - next blocks bottom y location
#
# The action space(2) 🎮:
# - Up (press accelerator) 
# - Do nothing (don't press accelerator) 
#
# The reward function 💰: 
# - For each vertical block it passes through it gains a positive reward of +1. Each time a terminal state reached it receives a negative reward of -1.

# %% [markdown] id="aV1466QP8crz"
# ### Define the new Policy 🧠
# - We need to have a deeper neural network since the environment is more complex

# %% id="I1eBkCiX2X_S"
class Policy(nn.Module):
    def __init__(self, s_size, a_size, h_size):
        super(Policy, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(s_size, h_size),
            nn.SiLU(),
            nn.Linear(h_size, h_size*2),
            nn.SiLU(),
            nn.Linear(h_size*2, a_size),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        # Define the forward process here
        return self.layers(x)
    
    def act(self, state):
        state = torch.from_numpy(np.asarray(state)).float().unsqueeze(0).to(device)
        probs = self.forward(state).cpu()
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)


# %% [markdown] id="47iuAFqV8Ws-"
# #### Solution

# %% [raw] id="wrNuVcHC8Xu7"
# class Policy(nn.Module):
#     def __init__(self, s_size, a_size, h_size):
#         super(Policy, self).__init__()
#         self.fc1 = nn.Linear(s_size, h_size)
#         self.fc2 = nn.Linear(h_size, h_size*2)
#         self.fc3 = nn.Linear(h_size*2, a_size)
#
#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return F.softmax(x, dim=1)
#     
#     def act(self, state):
#         state = torch.from_numpy(state).float().unsqueeze(0).to(device)
#         probs = self.forward(state).cpu()
#         m = Categorical(probs)
#         action = m.sample()
#         return action.item(), m.log_prob(action)

# %% [markdown] id="SM1QiGCSbBkM"
# ### Define the hyperparameters ⚙️
# - Because this environment is more complex.
# - Especially for the hidden size, we need more neurons.

# %% id="y0uujOR_ypB6"
DEFAULT_PARAMS = {
    "h_size": 64,
    "n_training_episodes": N_TIMESTEPS, #50000,
    "n_evaluation_episodes": 10,
    "max_t": 10000,
    "gamma": 0.99,
    "lr": 1e-4,
    "env_id": env_id,
    "state_space": s_size,
    "action_space": a_size,
}

# %%
from typing import Any, Dict
import torch
import torch.nn as nn
import optuna


# %%
def sample_params(trial: optuna.Trial) -> Dict[str, Any]:
    gamma = 1.0 - trial.suggest_float("1-gamma", 0.0001, 0.1, log=True)
    learning_rate = trial.suggest_float("lr", 1e-6, 1, log=True)
    h_size = trial.suggest_categorical("h_size", [32, 64, 128])
    
    return {
        "h_size": h_size,
        "gamma": gamma,
        "lr": learning_rate,
    }


# %% [markdown]
# ### Pruning Callback

# %% id="U5ijWTPzxSmd" vscode={"languageId": "python"}
from stable_baselines3.common.callbacks import EvalCallback

class TrialEvalCallback:
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
        policy,
        eval_env: gym.Env,
        trial: optuna.Trial,
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        deterministic: bool = True,
        verbose: int = 0,
    ):

        # super().__init__(
        #     eval_env=eval_env,
        #     n_eval_episodes=n_eval_episodes,
        #     eval_freq=eval_freq,
        #     deterministic=deterministic,
        #     verbose=verbose,
        # )
        self.policy = policy
        self.eval_env = eval_env
        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.trial = trial
        self.eval_idx = 0
        self.is_pruned = False
        self.n_calls = 0

    def evaluate(self):
        mean, std = evaluate_agent(
               self.eval_env, 
               DEFAULT_PARAMS["max_t"], 
               self.n_eval_episodes,
               self.policy)
        
        self.last_mean_reward = mean
        
        return mean, std
    
    def _on_step(self) -> bool:
        self.n_calls += 1
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Evaluate policy (done in the parent class)
            self.evaluate()
            self.eval_idx += 1
            # Send report to Optuna
            self.trial.report(self.last_mean_reward, self.eval_idx)
            # Prune trial if need
            if self.trial.should_prune():
                self.is_pruned = True
                return False
        return True


# %% [markdown] id="wyvXTJWm9GJG"
# ###  Train it
# - We're now ready to train our agent 🔥.

# %%
def objective(trial: optuna.Trial) -> float:
    pixelcopter_hyperparameters = DEFAULT_PARAMS.copy()
    pixelcopter_hyperparameters.update(sample_params(trial))

    pixelcopter_policy = Policy(pixelcopter_hyperparameters["state_space"], pixelcopter_hyperparameters["action_space"], pixelcopter_hyperparameters["h_size"]).to(device)
    pixelcopter_optimizer = optim.Adam(pixelcopter_policy.parameters(), lr=pixelcopter_hyperparameters["lr"])
    prune_cb = TrialEvalCallback(pixelcopter_policy, env, trial, eval_freq=EVAL_FREQ)
    
    scores = reinforce(env, pixelcopter_policy,
                       pixelcopter_optimizer,
                       pixelcopter_hyperparameters["n_training_episodes"], 
                       pixelcopter_hyperparameters["max_t"],
                       pixelcopter_hyperparameters["gamma"], 
                       1000,
                       reset_info=False,
                       callbacks=[prune_cb],
                      )
    
    
    if prune_cb.is_pruned:
        raise optuna.exceptions.TrialPruned()
        
    trial.set_user_attr("policy", pixelcopter_policy)
    trial.set_user_attr("params", pixelcopter_hyperparameters)
    
    return prune_cb.last_mean_reward

# %%

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

# %%
study.enqueue_trial({"1-gamma": 0.01, "lr": 1e-4, "h_size": 64})
study.enqueue_trial({"1-gamma": 5e-4, "lr": 5e-5, "h_size": 32})
study.enqueue_trial({"1-gamma": 0.01, "lr": 1e-4, "h_size": 64})

# %% id="4UU17YpjymPr" vscode={"languageId": "python"}
# %%time


try:
    study.optimize(objective, n_trials=N_TRIALS, n_jobs=N_JOBS, timeout=TIMEOUT)
except KeyboardInterrupt:
    pass

# %% [markdown] id="8kwFQ-Ip85BE"
# ### Publish our trained model on the Hub 🔥

# %%
best_trial = study.best_trial
pixelcopter_policy = best_trial.user_attrs["policy"]
pixelcopter_hyperparameters = best_trial.user_attrs["params"]

# %% id="6PtB7LRbTKWK"
repo_id = f"patonw/Reinforce-{env_id}" #TODO Define your repo id {username/Reinforce-{model-id}}
push_to_hub(repo_id,
                pixelcopter_policy, # The model we want to save
                pixelcopter_hyperparameters, # Hyperparameters
                eval_env, # Evaluation environment
                video_fps=30,
                reset_info=False,
                )

# %%
