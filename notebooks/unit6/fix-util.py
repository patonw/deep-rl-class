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

# %%
import sys
sys.path.append("../..")
from util import *

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

# %%
env_id = "PandaPickAndPlace-v3"
eval_env = gym.make(env_id, render_mode='rgb_array')
#  do not update them at test time
eval_env.training = False
# reward normalization is not needed at test time
eval_env.norm_reward = False

# %%
model_name = "tqc-PandaPickAndPlace-v3";
model = TQC.load(model_name, env=eval_env)

# %%
evaluate_agent(eval_env, 1_000, 10, lambda x: model.predict(x), reset_info=True)

# %%
record_video(eval_env, lambda x: model.predict(x), "replay.mp4", 10, reset_info=True, render_mode=None)

# %%
