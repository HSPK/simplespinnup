import numpy as np
import core
import gymnasium as gym
import torch
import torch.nn as nn


class GAEBuffer(object):
    def __init__(self, obs_dim, act_dim, size, info_shapes, gamma, lam):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.info_bufs = {
            k: np.zeros([size] + list(v), dtype=np.float32)
            for k, v in info_shapes.items()
        }
        self.info_sorted_keys = core.keys_as_sorted_list(self.info_bufs)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, logp, info):
        assert self.ptr < self.max_size
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        for i, k in enumerate(self.info_sorted_keys):
            self.info_bufs[k][self.ptr] = info[i]
        self.ptr += 1

    def finish_path(self, last_val=0):
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)

        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = core.discount_cumsum(deltas, self.gamma * self.lam)
        self.ret_buf[path_slice] = core.discount_cumsum(rews, self.gamma)[:-1]

        self.path_start_idx = self.ptr

    def get(self):
        assert self.ptr == self.max_size
        self.path_start_idx, self.ptr = 0, 0

        adv_mean, adv_std = np.mean(self.adv_buf), np.std(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std

        return [
            self.obs_buf,
            self.act_buf,
            self.adv_buf,
            self.ret_buf,
            self.logp_buf,
        ] + core.values_as_sorted_list(self.info_bufs)


def trpo(
    env_fn=lambda: gym.make("CarPole-v1"),
    actor_critic=core.MLPActorCritic,
    ac_kwargs=dict(),
    seed=0,
    epochs=50,
    max_steps=5000,
    max_ep_len=1000,
    gamm=0.99,
    lam=0.97,
    delta=0.01,
    vf_lr=1e-3,
    train_v_iters=80,
    cg_iters=10,
    backtrack_iters=10,
    backtrack_coeff=0.8,
):
    env = env_fn()
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape
    
    # Set random seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Create actor-critic module
    ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs)
    buf = GAEBuffer()
