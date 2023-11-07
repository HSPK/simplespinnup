import numpy as np
import torch.nn as nn
import scipy
import torch
from gymnasium.spaces import Box, Discrete, Space


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def keys_as_sorted_list(dict):
    return sorted(list(dict.keys()))


def values_as_sorted_list(dict):
    return [dict[k] for k in keys_as_sorted_list(dict)]


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input:
        vector x,
        [x0,
         x1,
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class Actor(nn.Module):
    def _distribution(self, obs):
        raise NotImplementedError

    def _log_prob_from_distribution(self, pi, act):
        raise NotImplementedError

    def forward(self, obs, act=None):
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a


class MLPActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit) -> None:
        super().__init__()
        pi_sizes = [obs_dim] + list(hidden_sizes) + [act_dim]
        self.pi = mlp(pi_sizes, activation, nn.Tanh)
        self.act_limit = act_limit

    def forward(self, obs):
        return self.pi(obs) * self.act_limit


class MLPQFunction(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation) -> None:
        super().__init__()
        self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs, act):
        q = self.q(torch.cat([obs, act], dim=-1))
        return torch.squeeze(q, -1)


class MLPQActorCritic(nn.Module):
    def __init__(
        self,
        observation_space,
        action_space,
        hidden_sizes=[256, 256],
        activation=nn.ReLU,
    ):
        super().__init__()
        obs_dim = observation_space.shape
        act_dim = action_space.shape[0]
        act_limit = action_space.hight[0]

        self.pi = MLPActor(obs_dim, act_dim, hidden_sizes, activation, act_limit)
        self.q = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)

    def act(self, obs):
        with torch.no_grad():
            return self.pi(obs).numpy()


class MLPQ2ActorCritic(nn.Module):
    def __init__(
        self,
        observation_space,
        action_space,
        hidden_sizes=[256, 256],
        activation=nn.ReLU,
    ):
        super().__init__()
        obs_dim = observation_space.shape
        act_dim = action_space.shape[0]
        act_limit = action_space.hight[0]

        self.pi = MLPActor(obs_dim, act_dim, hidden_sizes, activation, act_limit)
        self.q1 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)
        self.q2 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)

    def act(self, obs):
        with torch.no_grad():
            return self.pi(obs).numpy()


class MLPCategoricalActor(Actor):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.logits_net = mlp(
            sizes=[obs_dim] + list(hidden_sizes) + [act_dim], activation=activation
        )

    def _distribution(self, obs):
        logits = self.logits_net(obs)
        return torch.distributions.Categorical(logits=logits)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)


class MLPGuassianActor(Actor):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        self.mu_net = mlp(
            sizes=[obs_dim] + list(hidden_sizes) + [act_dim], activation=activation
        )

    def _distribution(self, obs):
        mu = self.mu_net(obs)
        std = torch.exp(self.log_std)
        return torch.distributions.Normal(mu, std)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act).sum(axis=-1)


class MLPCritic(nn.Module):
    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()
        self.v_net = mlp(
            sizes=[obs_dim] + list(hidden_sizes) + [1], activation=activation
        )

    def forward(self, obs):
        return self.v_net(obs).squeeze(dim=-1)


class MlpDklActorCritic(nn.Module):
    def __init__(
        self,
        observation_space: Space,
        action_space,
        hidden_sizes=[64, 64],
        activation=nn.Tanh,
    ):
        super().__init__()

        obs_dim = observation_space.shape[0]
        if isinstance(action_space, Box):
            self.pi = MLPGuassianActor(
                obs_dim, action_space.shape[0], hidden_sizes, activation
            )
        elif isinstance(action_space, Discrete):
            self.pi = MLPCategoricalActor(
                obs_dim, action_space.n, hidden_sizes, activation
            )
        self.v = MLPCritic(obs_dim, hidden_sizes, activation)

    def step(self, obs):
        with torch.no_grad():
            pi = self.pi._distribution(obs)
            a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            v = self.v(obs)
        return a.numpy(), v.numpy(), logp_a.numpy()


class MLPActorCritic(nn.Module):
    def __init__(
        self,
        observation_space: Space,
        action_space: Space,
        hidden_sizes=[64, 64],
        activation=nn.Tanh,
    ):
        super().__init__()

        obs_dim = observation_space.shape[0]
        if isinstance(action_space, Box):
            self.pi = MLPGuassianActor(
                obs_dim, action_space.shape[0], hidden_sizes, activation
            )
        elif isinstance(action_space, Discrete):
            self.pi = MLPCategoricalActor(
                obs_dim, action_space.n, hidden_sizes, activation
            )
        self.v = MLPCritic(obs_dim, hidden_sizes, activation)

    def step(self, obs):
        with torch.no_grad():
            pi = self.pi._distribution(obs)
            a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            v = self.v(obs)
        return a.numpy(), v.numpy(), logp_a.numpy()

    def act(self, obs):
        return self.step(obs)[0]


class GAEBuffer(object):
    def __init__(self, obs_dim: int, act_dim: int, size: int, gamma: float, lam: float):
        self.obs_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, val, rew, logp):
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.logp_buf[self.ptr] = logp
        self.val_buf[self.ptr] = val
        self.rew_buf[self.ptr] = rew
        self.ptr += 1

    def finish_path(self, last_val=0):
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
        assert self.ptr == self.max_size
        self.ptr, self.path_start_idx = 0, 0

        # advantage normalization trick
        adv_mean, adv_std = np.mean(self.adv_buf), np.std(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std

        ret = dict(
            obs=self.obs_buf,
            act=self.act_buf,
            ret=self.ret_buf,
            adv=self.adv_buf,
            logp=self.logp_buf,
        )
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in ret.items()}
