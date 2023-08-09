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

# %% id="HpiB8VdnQ7Bk"
import os

import gymnasium as gym
import panda_gym

from huggingface_sb3 import load_from_hub, package_to_hub

from stable_baselines3 import A2C
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from sb3_contrib.tqc import TQC

from huggingface_hub import notebook_login

import wandb
from wandb.integration.sb3 import WandbCallback

# %% id="zXzAu3HYF1WD"
env_id = "PandaPickAndPlace-v3"

# Create the env
env = gym.make(env_id)

# Get the state space and action space
s_size = env.observation_space
a_size = env.action_space

# %%
wandb_run = wandb.init(
    project=env_id,
    sync_tensorboard=True,
    monitor_gym=True,
)

# %% id="E-U9dexcF-FB"
print("_____OBSERVATION SPACE_____ \n")
print("The State Space is: ", s_size)
print("Sample observation", env.observation_space.sample()) # Get a random observation

# %% id="J-cC-Feg9iMm"
# 1 - 2
env_id = "PandaPickAndPlace-v3"

env = gym.make(env_id)

# 4
from stable_baselines3 import HerReplayBuffer, SAC
model = TQC(policy = "MultiInputPolicy",
            env = env,
            batch_size=2048,
            gamma=0.95,
            learning_rate=1e-4,
            train_freq=64,
            gradient_steps=64,
            tau=0.05,
            replay_buffer_class=HerReplayBuffer,
            replay_buffer_kwargs=dict(
                n_sampled_goal=4,
                goal_selection_strategy="future",
            ),
            policy_kwargs=dict(
                net_arch=[512, 512, 512],
                n_critics=2,
            ),
            tensorboard_log=f"runs/{wandb_run.id}",
           )

# 5
model.learn(1_000_000, progress_bar=True, callback=WandbCallback(verbose=2))
wandb_run.finish()

# %% id="-UnlKLmpg80p"
# 6
model_name = "tqc-PandaPickAndPlace-v3";
model.save(model_name)
# env.save("vec_normalize.pkl")

# %% id="-UnlKLmpg80p"
# 7
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Load the saved statistics
# eval_env = DummyVecEnv([lambda: gym.make("PandaPickAndPlace-v3")])
# eval_env = VecNormalize.load("vec_normalize.pkl", eval_env)

#  do not update them at test time
# eval_env.training = False
# reward normalization is not needed at test time
# eval_env.norm_reward = False

# Load the agent
model = TQC.load(model_name, env=env)

mean_reward, std_reward = evaluate_policy(model, env)

print(f"Mean reward = {mean_reward:.2f} +/- {std_reward:.2f}")

# %% id="-UnlKLmpg80p"
# 8
package_to_hub(
    model=model,
    model_name=f"tqc-{env_id}",
    model_architecture="TQC",
    env_id=env_id,
    eval_env=env,
    repo_id=f"patonw/tqc-{env_id}", # TODO: Change the username
    commit_message="Initial commit",
)

# %%
