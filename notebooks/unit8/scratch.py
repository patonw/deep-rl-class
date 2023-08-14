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
import time

from wasabi import Printer

import gymnasium as gym

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm.auto import tqdm

msg = Printer()

# %%
# %load_ext autoreload
# %autoreload 2
from lib import parse_args
import lib.hugs as hugs
import lib.ppo as ppo

# %%
args = parse_args()
run_name = 'foobar'

# %%
# Override default args if running in a juypter kernel
if '__session__' in dir():
    args.total_timesteps = 500_000
    args.repo_id = f'patonw/cleanppo-{args.env_id}'

# %%
print(f'Training agent with arguments:\n{vars(args)}')

# %% [raw]
# env_id = 'LunarLander-v2'
# hub_repo = hugs.ModelHubRepo(
#     repo_id=f'patonw/cleanppo-{env_id}',
#     model=nn.Sequential(),
#     hyperparams=args,
#     eval_env=None,
# )

# %%
vector_envs = ppo.get_vector_envs(args.env_id, args.num_envs, args.seed, args.capture_video, run_name)

# %%
agent = ppo.Agent(vector_envs)

# %%
trainer = ppo.Trainer(args, agent, vector_envs)

# %%
eval_env = gym.make(args.env_id)
hugs._evaluate_agent(eval_env, 10, agent, trainer.device)

# %%
trainer.fit(callbacks = [
    lambda pbar: pbar.set_postfix(
        global_step=trainer._global_step,
        eval_reward=hugs._evaluate_agent(eval_env, 10, agent, trainer.device),
    ),
])

# %%
hugs._evaluate_agent(eval_env, 100, agent, trainer.device)

# %%
