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
import gymnasium as gym
from gymnasium.wrappers import TransformObservation
from gymnasium.core import ObservationWrapper
from gymnasium.spaces.box import Box
import numpy as np
from einops import rearrange, reduce

from stable_baselines3.common.env_util import make_vec_env

# %%
import numpy as np
import esnpy as esn

# %%
ECHO_SIZE=64
env_id = "CartPole-v1"

# %%
env_raw = gym.make(env_id)

# %%
res_builder = esn.ReservoirBuilder(
    size=ECHO_SIZE,
    leaky=0.9,
    fn=np.tanh,
    input_size=env_raw.observation_space.shape[0] + (env_raw.action_space.shape or 1),
    input_init=esn.init.UniformDenseInit(-2, 2),
    intern_init=esn.init.NormalSparseInit(0, 1, density=0.01),
    intern_tuners=[esn.tune.SpectralRadiusTuner(1.3)],
)

# %%
np.random.seed(0)

# %%
OBS_MASK = np.array([0, 1, 0, 1])
tb_log_name=f"esn{ECHO_SIZE}-nopos"
# tb_log_name=f"nopos"

# %%
class EchoWrapper(ObservationWrapper):
    def __init__(self, env, *, echo_seed=42):
        super().__init__(env)
        self._last_action = env.action_space.sample()
        self._echo_seed = echo_seed
        self._reservoir = None
        obs_space = env.observation_space
        
        self.observation_space = Box(
            np.append(obs_space.low, [-100.0] * ECHO_SIZE),
            np.append(obs_space.high, [100.0] * ECHO_SIZE),
            (obs_space.shape[0] + ECHO_SIZE,),
            obs_space.dtype,
            echo_seed,
        )
        
    def reset(self, **kwargs):
        self._reservoir = res_builder.build(self._echo_seed)
        obs, info = super().reset(**kwargs)
        # obs = np.append(obs, self._reservoir._state)
        return obs, info
    
    def step(self, action):
        self._last_action = action
        return super().step(action)
        
    def observation(self, observation):
        # Action feedback seems to have no effect
        X = np.append(observation, [self._last_action])
        
        # This ESN implementation seems to have trouble with NaNs...
        echo = np.nan_to_num(self._reservoir(rearrange(X, 'x -> 1 x')))
        return np.append(observation, echo)


# %%
env_wrap = TransformObservation(gym.make(env_id), lambda obs: np.where(OBS_MASK, obs, 0))
env_wrap = EchoWrapper(env_wrap)

envs = make_vec_env(
    lambda: EchoWrapper(TransformObservation(gym.make(env_id), lambda obs: np.where(OBS_MASK, obs, 0))),
    n_envs=4,
)

# %%
env_wrap.observation_space

# %%
env_raw.reset(seed=42), env_wrap.reset(seed=42)

# %%
env_raw.observation_space, env_wrap.observation_space

# %%
# %load_ext autoreload
# %autoreload 2
from lib import parse_args
import lib.hugs as hugs
import lib.ppo as ppo

# %%
args = parse_args()

# Override default args if running in a juypter kernel
if '__session__' in dir():
    args.total_timesteps = 500_000
    args.repo_id = f'patonw/cleanppo-{args.env_id}'

# %% [raw]
# vector_envs = gym.vector.SyncVectorEnv([
#     lambda: EchoWrapper(TransformObservation(gym.make(env_id), lambda obs: np.where(OBS_MASK, obs, 0))) for i in range(args.num_envs)]
# )
# agent = ppo.Agent(vector_envs)
#
# trainer = ppo.Trainer(args, agent, vector_envs)

# %%
from huggingface_hub import HfApi, upload_folder
from huggingface_hub.repocard import metadata_eval_result, metadata_save

from pathlib import Path
import torch
import datetime
import tempfile
import json
import shutil
import imageio
import numpy as np

import wasabi

from typing import Any, Union
from dataclasses import dataclass
from argparse import Namespace
from textwrap import indent, dedent, wrap

def _evaluate_agent(env, n_eval_episodes, policy, device):
    """
    Evaluate the agent for ``n_eval_episodes`` episodes and returns average reward and std of reward.
    :param env: The evaluation environment
    :param n_eval_episodes: Number of episode to evaluate the agent
    :param policy: The agent
    """
    episode_rewards = []
    for episode in range(n_eval_episodes):
        state, info = env.reset()
        step = 0
        done = False
        total_rewards_ep = 0

        while done is False:
            state = torch.Tensor(state).to(device)
            action, _, _, _ = policy.get_action_and_value(state)
            new_state, reward, done, trunc, info = env.step(action.cpu().numpy())
            total_rewards_ep += reward
            if done:
                break
            state = new_state
        episode_rewards.append(total_rewards_ep)
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)

    return mean_reward, std_reward


# %% [raw]
# trainer.fit(callbacks = [
#     lambda pbar: pbar.set_postfix(
#         global_step=trainer._global_step,
#         eval_reward=hugs._evaluate_agent(env_wrap, 10, agent, trainer.device),
#     ),
# ])

# %% [markdown]
# ## SB3

# %%
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

# %%
model = PPO(
    policy="MlpPolicy",
    env=envs,
    n_steps=1024,
    batch_size=64,
    n_epochs=4,
    gamma=0.999,
    gae_lambda=0.98,
    ent_coef=0.01,
    verbose=0,
    tensorboard_log="./runs/",
)

# %%
model.learn(
    total_timesteps=2_000_000,
    progress_bar=True,
    tb_log_name=tb_log_name,
)

# Save the model
model_name = f"esn_ppo-Masked{env_id}"
# model.save(model_name)

# %%
eval_env = Monitor(env_wrap)
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=True)
print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")

# %% [markdown]
# ## Findings

# %% [markdown]
# ![image.png](attachment:c725853b-0d03-436c-990a-d3c973b7713b.png)

# %% [markdown]
# - (Grey) Vanilla PPO baseline (grey) performs well but suffers from bouts of forgetfulness
# - (Yellow) Masking linear & angular velocity severely inhibits performance of vanilla PPO
# - (Green) With ESN, agent performs comparably with vanilla PPO
# - (Orange) Even with ESN on masked velocities and linear position agent has moderate trouble
# - (Cyan) Masking only position and angle, agent with ESN performs better than vanilla PPO and is more stable
# - (Pink) Vanilla PPO on masked position and angle suffers slight performance degradation
#
# ESN based agents above use reservoir of 256 nodes
#
# ESN using only velocities (masking position & angle) outperforms everything, including scenario masking velocity. This sounds reasonable since ESNs should excel at time integrals.

# %%
