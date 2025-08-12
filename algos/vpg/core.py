import numpy as np
from Module import Module, Identity, Sequential, Linear, Tanh
from Parameter import Parameter
from distributions.Categorical import Categorical
from distributions.Normal import Normal
from gymnasium.spaces import Box, Discrete
from Tensor import Tensor
from Tensor import no_grad

def combined_shape(length, shape=None):
  if shape is None:
    return (length,)
  return (length, shape) if np.isscalar(shape) else (length, *shape)

def manual_seed(seed):
  np.random.seed(seed)

def count_vars(module):
  return sum([np.prod(p.data.shape) for p in module.parameters()])

def mlp(sizes, activation, output_activation=Identity):
  layers = []
  for j in range(len(sizes)-1):
    act = activation if j < len(sizes)-2 else output_activation
    layers += [Linear(sizes[j], sizes[j+1]), act()]  # create layers with i/o feature size with activation in between
  return Sequential(*layers)  # pass it to Sequential to combine these layers

def discount_cumsum(x, discount):
  """
  Computes the discounted cumulative sum of a vector. (In prod use scipy)

  For a vector x = [x0, x1, x2], the output is:
  [x0 + discount * x1 + discount^2 * x2,
   x1 + discount * x2,
   x2]
  """
  result = np.zeros_like(x, dtype=float)
  running_sum = 0
  for t in reversed(range(len(x))):
    running_sum = x[t] + discount * running_sum
    result[t] = running_sum
  return result

class Actor(Module):
  def _distribution(self, obs):
    raise NotImplementedError

  def _log_prob_from_distribution(self, pi, act):
    raise NotImplementedError

  def forward(self, obs, act=None):
    # Produce action distributions for given observations, and optionally
    # compute the log likelihood of given actions under those distributions
    pi = self._distribution(obs)
    logp_a = None
    if act is not None:
      logp_a = self._log_prob_from_distribution(pi, act)
    return pi, logp_a
  
class MLPCategoricalActor(Actor):
  def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
    super().__init()
    self.logits_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

  def _distribution(self, obs):
    logits = self.logits_net(obs)
    return Categorical(logits=logits)  # create prob distribution

  def _log_prob_from_distribution(self, pi, act):
    return pi.log_prob(act)  # get log prob of categorical distribution
  
class MLPGaussianActor(Actor):
  def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
    super().__init__()
    log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
    self.log_std = Parameter(Tensor(log_std))
    # print(log_std)
    # print(Tensor(log_std))
    # print(self.log_std)
    self.mu_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

  def _distribution(self, obs):
    mu = self.mu_net(obs)
    std = Tensor(np.exp(self.log_std.data))  # convert log std to std
    return Normal(mu, std)

  def _log_prob_from_distribution(self, pi, act):
    return pi.log_prob(act).sum(axis=-1)    # Last axis sum needed for Torch normal distribution
  
class MLPCritic(Module):
  def __init__(self, obs_dim, hidden_sizes, activation):
    super().__init__()
    self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation)

  def forward(self, obs):
    # print(self.v_net(obs))
    return self.v_net(obs).squeeze(-1)
  
class MLPActorCritic(Module):
  def __init__(self, observation_space, action_space, hidden_sizes=(64,64), activation=Tanh):
    super().__init__()

    obs_dim = observation_space.shape[0]

    # policy builder depends on action space
    if isinstance(action_space, Box):
      # print(action_space.shape[0])
      self.pi = MLPGaussianActor(obs_dim, action_space.shape[0], hidden_sizes, activation)
    elif isinstance(action_space, Discrete):
      # print(action_space.n)
      self.pi = MLPCategoricalActor(obs_dim, action_space.n, hidden_sizes, activation)

    # build value function
    self.v = MLPCritic(obs_dim, hidden_sizes, activation)

  def step(self, obs):   # this is for the training phase where we want to store every param
    with no_grad():
      pi = self.pi._distribution(obs)
      a = pi.sample()
      logp_a = self.pi._log_prob_from_distribution(pi, a)
      v = self.v(obs)
    return a.numpy(), v.numpy(), logp_a.numpy()  # returns everything

  def act(self, obs):   # this is for the evaluation phase
    return self.step(obs)[0]   # only returns action