from argparse import Namespace
from collections import namedtuple
from dataclasses import dataclass
import random
import time

import numpy as np
import gymnasium as gym
from einops import rearrange, reduce

import torch
from .check import ShapeChecker

from torch import nn, tensor, Tensor
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

from tqdm.auto import tqdm

def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        #env.seed(seed) # TODO
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk()
  
def get_vector_envs(env_id, num_envs, seed, capture_video, run_name):
  vector_envs = gym.vector.SyncVectorEnv([lambda: make_env(env_id, seed + i, i, capture_video, run_name) for i in range(num_envs)])
  return vector_envs

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
  nn.init.orthogonal_(layer.weight, std)
  nn.init.constant_(layer.bias, bias_const)
  return layer

AgentResult = namedtuple('AgentResult', ['action', 'log_prob', 'entropy', 'value'])

class Agent(nn.Module):
  def __init__(self, envs):
    super().__init__()
    self.critic = nn.Sequential(
      layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
      nn.Tanh(),
      layer_init(nn.Linear(64, 64)),
      nn.Tanh(),
      layer_init(nn.Linear(64, 1), std=1.0),
    )
    
    self.actor = nn.Sequential(
      layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
      nn.Tanh(),
      layer_init(nn.Linear(64, 64)),
      nn.Tanh(),
      layer_init(nn.Linear(64, envs.single_action_space.n), std=0.01),
    )
    
  def get_value(self, x):
    return self.critic(x)
  
  def get_action_and_value(self, x, action=None):
    logits = self.actor(x)
    probs = Categorical(logits=logits)
    if action is None:
      action = probs.sample()
    return AgentResult(action, probs.log_prob(action), probs.entropy(), self.critic(x))

History = namedtuple("History", ['obs', 'actions', 'logprobs', 'rewards', 'dones', 'values'])
Batch = namedtuple("Batch", ["obs", "actions", "logprobs", "advantages", "returns", "values"])

class Trainer:
  args: Namespace
  agent: Agent
  envs: gym.vector.VectorEnv
  device: torch.device
  optimizer: torch.optim.Optimizer
  run_name: str
  _global_step: int = 0
  _writer: SummaryWriter
  _shaper: ShapeChecker
  _history: History = None

  def __init__(self, args, agent, envs):
    self.args = args
    self.device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    self.agent = agent.to(self.device)
    self.envs = envs
    
    self.optimizer = optim.Adam(self.agent.parameters(), lr=args.learning_rate, eps=1e-5)
    self.run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    self._writer = SummaryWriter(f'runs/{self.run_name}')
    self._shaper = ShapeChecker()
  
  def _reset(self):
    seed = self.args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = self.args.torch_deterministic
    
    self.agent = self.agent.to(self.device)
    self._history = None
    self._global_step = 0
    self._start_time = time.time()
    
  def history(self):
    if self._history is None:
      hist_shape = (self.args.num_steps, self.args.num_envs)
      obs_shape = hist_shape + self.envs.single_observation_space.shape
      action_shape = hist_shape + self.envs.single_action_space.shape
      
      # eh? Why keep history in VRAM during rollout, bouncing back and forth between CPU & GPU?
      # We can just move it once in bulk at the start of training.
      self._history = History(
        torch.zeros(obs_shape),
        torch.zeros(action_shape),
        torch.zeros(hist_shape),
        torch.zeros(hist_shape),
        torch.zeros(hist_shape, dtype=torch.bool),
        torch.zeros(hist_shape),
      )
      
    return self._history
  
  def _get_lr(self, eon, max_eon):
    if (not self.args.anneal_lr):
      return self.args.learning_rate
    
    progress = 1.0 - (eon - 1.0) / max_eon
    return progress * self.args.learning_rate
    
    
  def _summarize_step(self, info):
    # TODO!
    pass
  
  def _rollout(self, next_obs: Tensor, next_done: Tensor):
    history = self.history()
    
    # ???
    next_obs = torch.Tensor(self.envs.reset()[0])
    next_done = torch.zeros(self.args.num_envs)
    
    for step in range(self.args.num_steps):
      # next_obs = tensor(next_obs)
      
      self._global_step += self.args.num_envs
      history.obs[step] = next_obs
      history.dones[step] = next_done
      
      with torch.no_grad():
        action, logprob, _, value = self.agent.get_action_and_value(next_obs.to(self.device))
        
      history.values[step] = value.flatten()
      history.actions[step] = action
      history.logprobs[step] = logprob
      
      next_obs, reward, done, trunc, info = self.envs.step(action.cpu().numpy())
      self._shaper(reward, 'n', n=self.args.num_envs)
      history.rewards[step] = torch.tensor(reward).view(-1) # ? is this actually reshaping anything ?
      next_obs = tensor(next_obs)
      next_done = tensor(done | trunc)
      
      self._summarize_step(info)
      
    return next_obs, next_done

  def _calc_returns(self, next_obs, next_done):
    args = self.args
    history = self.history()
    rewards = history.rewards
    values = history.values
    dones = history.dones
    
    with torch.no_grad():
      next_value = self.agent.get_value(next_obs.to(self.device)).cpu()
      next_value = rearrange(next_value, 'n 1 -> 1 n')

      if args.gae:
          advantages = torch.zeros_like(rewards)
          lastgaelam = 0
          for t in reversed(range(args.num_steps)):
              if t == args.num_steps - 1:
                  nextnonterminal = ~next_done
                  nextvalues = next_value
              else:
                  nextnonterminal = ~dones[t + 1]
                  nextvalues = values[t + 1]
              delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
              advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
          returns = advantages + values
      else:
          returns = torch.zeros_like(rewards)
          for t in reversed(range(args.num_steps)):
              if t == args.num_steps - 1:
                  nextnonterminal = ~next_done
                  next_return = next_value
              else:
                  nextnonterminal = ~dones[t + 1]
                  next_return = returns[t + 1]
              returns[t] = rewards[t] + args.gamma * nextnonterminal * next_return
          advantages = returns - values
        
    return returns, advantages
  
  def _prepare_batch(self, returns, advantages):
    history = self.history()
    obs = rearrange(history.obs, 't n s -> (t n) s')
    actions = rearrange(history.actions, 't n -> (t n)')
    logprobs = rearrange(history.logprobs, 't n -> (t n)')
    values = rearrange(history.values, 't n -> (t n)')
    returns = rearrange(returns, 't n -> (t n)')
    advantages = rearrange(advantages, 't n -> (t n)')
    
    return Batch(obs, actions, logprobs, advantages, returns, values)
    
  def _approx_kl_div(self, mb, newlogprob):
    clip_coef = self.args.clip_coef
    clipfracs = []
    with torch.no_grad():
      logratio = newlogprob - mb.logprobs
      ratio = logratio.exp()
      old_kld = (-logratio).mean()
      kld = ((ratio - 1) - logratio).mean()
      clipfracs += [((ratio - 1.0).abs() > clip_coef).float().mean().item()]
    return kld, old_kld, clipfracs
  
  def _calc_policy_gain(self, mb, newlogprob):
    clip_coef = self.args.clip_coef
    
    logratio = newlogprob - mb.logprobs
    ratio = logratio.exp()
    

    if self.args.norm_adv:
      advantages = advantages = (mb.advantages - mb.advantages.mean()) / (mb.advantages.std() + 1e-8)
    else:
      advantages = mb.advantages
      
    raw_gain = ratio * advantages
    clip_gain = torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef) * advantages
    gain = torch.min(raw_gain, clip_gain)
    return gain.mean()
  
  def _calc_value_loss(self, mb, newvalue):
    clip_coef = self.args.clip_coef
    # self._shaper(newvalue, 'm 1', m=self.args.minibatch_size)
    newvalue = rearrange(newvalue, 'm ()-> m')
    raw_loss = (newvalue - mb.returns)**2
    if self.args.clip_vloss:
      delta = newvalue - mb.values
      clip_values = mb.values + torch.clamp(delta, -clip_coef, clip_coef)
      clip_loss = (clip_values - mb.returns) ** 2
      vloss = torch.max(raw_loss, clip_loss)
    else:
      vloss = raw_loss
    
    vloss = 0.5 * vloss.mean()
    return vloss
    
    
  def _do_minibatch(self, mb):
    ent_coef = self.args.ent_coef
    vf_coef = self.args.vf_coef
    _, newlogprob, entropy, newvalue = self.agent.get_action_and_value(mb.obs, mb.actions)
      
    loss = vf_coef * self._calc_value_loss(mb, newvalue)
    loss -= ent_coef * entropy.mean()
    loss -= self._calc_policy_gain(mb, newlogprob)
    
    self.optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(self.agent.parameters(), self.args.max_grad_norm)
    self.optimizer.step()
    
    return self._approx_kl_div(mb, newlogprob)

  def _do_epoch(self, batch):
    args = self.args
    all_idx = np.random.permutation(args.batch_size)
    num_chunks = args.batch_size // args.minibatch_size
    for idx in np.array_split(all_idx, num_chunks):
      minibatch = Batch(*[it[idx].to(self.device) for it in batch])
      
      kld, old_kld, clipfracs = self._do_minibatch(minibatch)
      
    writer = self._writer
    global_step = self._global_step
    
    writer.add_scalar("losses/old_approx_kl", old_kld.item(), global_step)
    writer.add_scalar("losses/approx_kl", kld.item(), global_step)
    writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
    
    # This doesn't make sense. Only use the kl div of the last minibatch?
    if args.target_kl is not None:
      if kld > args.target_kl:
        return False
      
    return True
    
  def fit(self, callbacks = []):
    args = self.args
    optimizer = self.optimizer
    writer = self._writer
    
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(self.args).items()])),
    )
    
    self._reset()
    
    next_obs = torch.Tensor(self.envs.reset()[0])
    next_done = torch.zeros(self.args.num_envs)
    max_eon = args.total_timesteps // args.batch_size
    
    for eon in (progress := tqdm(range(1, max_eon+1))):
      optimizer.param_groups[0]["lr"] = self._get_lr(eon, max_eon)
      
      next_obs, next_done = self._rollout(next_obs, next_done)
      
      returns, advantages = self._calc_returns(next_obs, next_done)
      
      batch = self._prepare_batch(returns, advantages)
      
      if not self._do_epoch(batch):
        break
        
      y_pred, y_true = batch.values.cpu().numpy(), batch.returns.cpu().numpy()
      var_y = np.var(y_true)
      explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
      # print(f"Explained variance: {explained_var}")

      global_step = self._global_step
      # TRY NOT TO MODIFY: record rewards for plotting purposes
      writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
      # writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
      # writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
      # writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
      # writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
      # writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
      # writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
      writer.add_scalar("losses/explained_variance", explained_var, global_step)
      # print("SPS:", int(global_step / (time.time() - self._start_time)))
      writer.add_scalar("charts/SPS", int(global_step / (time.time() - self._start_time)), global_step)
      
      for cb in callbacks:
        cb(progress)

    self.envs.close()
    self._writer.close()