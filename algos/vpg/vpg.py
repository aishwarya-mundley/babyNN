import time
import numpy as np
import gymnasium as gym
from algos.vpg.core import combined_shape, manual_seed, count_vars, MLPActorCritic, discount_cumsum
from blocks.Tensor import Tensor, no_grad
from blocks.Optimizer import Adam
import argparse
import wandb

class VPGBuffer:
  """
  A buffer for storing trajectories experienced by VPG agent interacting with
  env, and using Generalized Advantage Estimation (GAE-Lambda) for calculating
  the advantages of state-action pairs
  """

  def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
    self.obs_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
    self.act_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
    self.adv_buf = np.zeros(size, dtype=np.float32)
    self.rew_buf = np.zeros(size, dtype=np.float32)
    self.ret_buf = np.zeros(size, dtype=np.float32)
    self.val_buf = np.zeros(size, dtype=np.float32)
    self.logp_buf = np.zeros(size, dtype=np.float32)
    self.gamma, self.lam = gamma, lam
    self.ptr, self.path_start_idx, self.max_size = 0, 0, size

  def store(self, obs, act, rew, val, logp):
    """
    Append one timestep of agent-environment interaction to the buffer.
    """
    assert self.ptr < self.max_size     # buffer has to have room so you can store
    self.obs_buf[self.ptr] = obs
    self.act_buf[self.ptr] = act
    self.rew_buf[self.ptr] = rew
    self.val_buf[self.ptr] = val.item()
    self.logp_buf[self.ptr] = logp.item()
    self.ptr += 1

  def finish_path(self, last_val=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.

        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)
        
        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = discount_cumsum(deltas, self.gamma * self.lam)
        
        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1]
        
        self.path_start_idx = self.ptr

  def get(self):
    """
    Call this at the end of an epoch to get all of the data from
    the buffer, with advantages appropriately normalized (shifted to have
    mean zero and std one). Also, resets some pointers in the buffer.
    """
    assert self.ptr == self.max_size  # buffer has to be full before you can get
    self.ptr, self.path_start_idx = 0, 0
    # next two lines for advantage normalization - this is needed because raw adv can
    # be as small as -1, 1 and as large as -500, 500 which can make training unstable
    # as loss is directly proportional to adv
    adv_mean = sum(self.adv_buf)/len(self.adv_buf)
    adv_std = np.std(self.adv_buf)
    self.adv_buf = (self.adv_buf - adv_mean)/adv_std
    data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf,
                adv=self.adv_buf, logp=self.logp_buf)
    return {k: Tensor(np.array(v, dtype=np.float32), requires_grad=False) for k,v in data.items()}
  
def vpg(env_fn, actor_critic=MLPActorCritic, ac_kwargs=dict(), seed=0,
        steps_per_epoch=4000, epochs=50, gamma=0.99, pi_lr=3e-4, vf_lr=1e-3,
        train_v_iters=80, lam=0.97, max_ep_len=1000, save_freq=10):
  """
  Vanilla Policy Gradient with GAE-lambda for advantage estimation
  """

  # seeding is needed for reproducibility; where every time script is run, the env
  # resets, initial model weights and sequence of actions will be identical leading
  # to same final results
  manual_seed(seed)
  np.random.seed(seed)

  # Instantiate env
  env = env_fn()
  obs_dim = env.observation_space.shape
  act_dim = env.action_space.shape

  # Create actor-critic module
  ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs)

  # count variables
  var_counts = tuple(count_vars(module) for module in [ac.pi, ac.v])
  print("Number of parameters: ", var_counts)

  # setup experience buffer
  buf = VPGBuffer(obs_dim, act_dim, steps_per_epoch, gamma, lam)

  # setup function for computing vpg policy loss
  def compute_loss_pi(data):
    obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']

    # Policy loss
    pi, logp = ac.pi(obs, act)
    loss_pi = -(logp * adv).mean()

    # extra info
    approx_kl = (logp_old - logp).mean().item()
    ent = pi.entropy().mean().item()
    pi_info = dict(kl=approx_kl, ent=ent)

    return loss_pi, pi_info

  # setup function for computing value loss
  def compute_loss_v(data):
    obs, ret = data['obs'], data['ret']
    return ((ac.v(obs) - ret)**2).mean()

  # setup optimizers for policy and value function
  pi_optimizer = Adam(ac.pi.parameters(), lr=pi_lr)
  vf_optimizer = Adam(ac.v.parameters(), lr=vf_lr)

  wandb.init(
    project="vpg-halfcheetah",  # Your project name
    config={
        "env": "HalfCheetah-v4",
        "algo": "VPG",
        "gamma": 0.99,
        "pi_lr": 3e-4,
        "vf_lr": 1e-3,
        "epochs": 50,
        "steps_per_epoch": 4000
    }
)

  def update():
    data = buf.get()
    
    # Get old policy distribution
    with no_grad():
      pi_old, _ = ac.pi(data['obs'], data['act'])

    # get loss and info values before update
    pi_l_old, pi_info_old = compute_loss_pi(data)
    pi_l_old = pi_l_old.item()
    v_l_old = compute_loss_v(data).item()

    # train policy with single step of gradient decent - this is 1 step
    pi_optimizer.zero_grad()
    loss_pi, pi_info = compute_loss_pi(data)
    loss_pi.backward()
    pi_optimizer.step()

    # Get new policy distribution after the update
    with no_grad():
      pi_new, _ = ac.pi(data['obs'], data['act'])
    # Calculate the KL divergence between the old and new policies
    # D_KL(pi_old || pi_new)
    kl = (pi_old.log_prob(data['act']) - pi_new.log_prob(data['act'])).mean().item()
    ent = pi_new.entropy().mean().item() # Use entropy from the new policy

    # value function learning - this is for {train_v_iter} no of step
    for i in range(train_v_iters):
      vf_optimizer.zero_grad()
      loss_v = compute_loss_v(data)
      loss_v.backward()
      vf_optimizer.step()

    # Re-calculate losses with the updated parameters to get a meaningful delta
    loss_pi_new, _ = compute_loss_pi(data)
    loss_v_new = compute_loss_v(data)

    # Log changes from update
    print("\nLossPi: ",pi_l_old)
    print("LossV: ", v_l_old)
    print("KL: ", kl)
    print("Entropy: ", ent)
    print("DeltaLossPi: ", loss_pi_new.item() - pi_l_old)
    print("DeltaLossV: ", loss_v_new.item() - v_l_old)
    wandb.log({
        "loss_pi": loss_pi_new.item(),
        "loss_v": loss_v_new.item(),
        "kl": kl,
        "entropy": ent,
        "mean_adv": data['adv'].data.mean().item(),
        "train_return": data['ret'].data.mean().item()
    })

  # Prepare for interaction with env
  state_time = time.time()
  o, ep_ret, ep_len = env.reset(), 0, 0
  if isinstance(o, (tuple)):
    o= o[0]  # if env returns tuple, take first element (e.g. obs, info)
  if isinstance(o, (dict)):
    o_flat = np.concatenate([np.asarray(oi, dtype=np.float32).flatten() for oi in o.values()])
  elif isinstance(o, (list, tuple)):
    o_flat = np.concatenate([np.asarray(oi, dtype=np.float32).flatten() for oi in o])
  else:
    o_flat = np.asarray(o, dtype=np.float32)
  if o_flat.ndim == 1:
    o_flat = o_flat.reshape(1, -1)  # shape (1, 17)

  # Main loop : collect experience in env and update/log each epoch
  for epoch in range(epochs):
    for t in range(steps_per_epoch):
      a, v, logp = ac.step(Tensor(o_flat))     # actor gives action and its log prob, critic gives value
      if isinstance(a, Tensor):
        a = a.data
      a = np.asarray(a).squeeze()
      if a.ndim == 2 and a.shape[0] == 1:
        a = a[0]  # shape (6,)
      elif a.ndim > 1 and a.shape[0] == steps_per_epoch:
        a = a[t]  # get the action for current timestep

      next_o, r, terminated, truncated, _ = env.step(a)    # take that action in env and get reward r and next_obs
      done = terminated or truncated
      if isinstance(next_o, tuple):
        next_o = next_o[0]
      if isinstance(next_o, dict):
        next_o_flat = np.concatenate([np.asarray(oi, dtype=np.float32).flatten() for oi in next_o.values()])
      elif isinstance(next_o, (tuple, list)):
        next_o_flat = np.concatenate([np.asarray(oi, dtype=np.float32).flatten() for oi in next_o])
      else:
        next_o_flat = np.asarray(next_o, dtype=np.float32)
      if next_o_flat.ndim == 1:
        next_o_flat = next_o_flat.reshape(1, -1)
      ep_ret += r
      ep_len += 1

      # save and log
      buf.store(o_flat, a, r, v, logp)  # store current obs, action, reward, value and log prob in buffer

      # update obs (critical!)
      o_flat = next_o_flat

      timeout = ep_len == max_ep_len
      terminal = done or timeout
      epoch_ended = t == steps_per_epoch-1

      if terminal or epoch_ended:
        if epoch_ended and not(terminal):
          print('Warning: trajectory cut off by epoch at %d steps.'%ep_len, flush=True)

        # if trajectory didn't reach terminal state, bootstrap value target bcoz since it was
        # cut short in middle it is not true terminal state hence we don't know future rewards
        # to solve this we ask critic for value estimate of last state the agent saw, this value
        # this acts as stand-in for all future rewards we couldn't see
        if timeout or epoch_ended:
          _, v, _ = ac.step(Tensor(o_flat))
        else:
          v = 0
        buf.finish_path(v)
        # if terminal:
        #   # only save EpRet / EpLen if trajectory finished
        o, ep_ret, ep_len = env.reset(), 0, 0

    # perform VPG update
    update()

parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, default='HalfCheetah-v5')
parser.add_argument('--hid', type=int, default=128)
parser.add_argument('--l', type=int, default=3)
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--seed', '-s', type=int, default=0)
parser.add_argument('--steps', type=int, default=4000)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--exp_name', type=str, default='vpg')
args = parser.parse_args()

vpg(lambda: gym.make(args.env), actor_critic=MLPActorCritic,
    ac_kwargs=dict(hidden_sizes=[args.hid]*args.l), gamma=args.gamma,
    seed=args.seed, steps_per_epoch=args.steps, epochs=args.epochs)