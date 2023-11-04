from core import discount_cumsum, combined_shape, count_vars
import torch.nn as nn
from gymnasium import Env
import core
import gymnasium as gym
import numpy as np
import torch
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


class VPGBuffer(object):
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


def vpg(
    env_fn,
    actor_critic,
    ac_kwargs: dict,
    epoch: int,
    max_steps: int,
    max_ep_len: int,
    train_v_iters: int,
    pi_lr: float,
    vf_lr: float,
    gamma: float,
    lam: float,
):
    env = env_fn()
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape
    logger.info(f"Observation space: {obs_dim}, Action space: {act_dim}")

    ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs)
    buf = VPGBuffer(obs_dim, act_dim, max_steps, gamma, lam)

    var_counts = tuple(count_vars(module) for module in [ac.pi, ac.v])
    print("\nNumber of parameters: \t pi: %d, \t v: %d\n" % var_counts)
    optimizer_pi = torch.optim.Adam(ac.pi.parameters(), lr=pi_lr)
    optimizer_v = torch.optim.Adam(ac.v.parameters(), lr=vf_lr)

    def compute_loss_pi(data):
        obs, act, adv = data["obs"], data["act"], data["adv"]
        _, logp = ac.pi(obs, act)
        loss_pi = -(logp * adv).mean()
        return loss_pi

    def compute_loss_v(data):
        obs, ret = data["obs"], data["ret"]
        return ((ac.v(obs) - ret) ** 2).mean()

    def update():
        data = buf.get()

        optimizer_pi.zero_grad()
        loss_pi = compute_loss_pi(data)
        loss_pi.backward()
        optimizer_pi.step()

        for i in range(train_v_iters):
            optimizer_v.zero_grad()
            loss_v = compute_loss_v(data)
            loss_v.backward()
            optimizer_v.step()

        return loss_pi.item(), loss_v.item()

    (o, o_info), ep_ret, ep_len = env.reset(), 0, 0
    best_reward = -np.inf
    for epoch in range(epoch):
        ep_rews = 0
        episodes = 0
        for t in range(max_steps):
            a, v, logp = ac.step(torch.as_tensor(o, dtype=torch.float32))
            next_o, r, d, tructated, _ = env.step(a)
            ep_ret += r
            ep_len += 1
            ep_rews += r

            buf.store(o, a, v, r, logp)
            o = next_o

            timeout = ep_len == max_ep_len
            terminal = d or tructated or timeout

            epoch_ended = t == max_steps - 1
            if terminal or epoch_ended:
                if epoch_ended and not terminal:
                    print(
                        "Warning: trajectory cut off by epoch at %d steps." % ep_len,
                        flush=True,
                    )
                last_val = (
                    0
                    if d or tructated
                    else ac.step(torch.as_tensor(o, dtype=torch.float32))[1].item()
                )
                buf.finish_path(last_val)
                (o, o_info), ep_ret, ep_len = env.reset(), 0, 0
                episodes += 1

        loss_pi, loss_v = update()
        ep_rews /= episodes
        if ep_rews > best_reward:
            best_reward = ep_rews
            torch.save(ac.state_dict(), "vpg.pt")
        logger.info(
            f"Epoch: {epoch}, Loss pi: {loss_pi}, Loss v: {loss_v}, Reward: {ep_rews}"
        )


def vpg_test(env_fn, actor_critic, ac_kwargs, test_epochs, model_path):
    env = env_fn()
    ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs)
    ac.load_state_dict(torch.load(model_path))
    ac.eval()

    for epoch in range(test_epochs):
        (o, o_info), ep_ret, ep_len = env.reset(), 0, 0
        while True:
            a, v, logp = ac.step(torch.as_tensor(o, dtype=torch.float32))
            next_o, r, d, tructated, _ = env.step(a)
            ep_ret += r
            ep_len += 1
            o = next_o
            terminal = d or tructated
            if terminal:
                print(f"Epoch: {epoch}, Reward: {ep_ret}")
                (o, o_info), ep_ret, ep_len = env.reset(), 0, 0
                break
        cmd = input("Continue? [y/n]")
        if cmd == "n":
            break


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    if args.test:
        vpg_test(
            lambda: gym.make("LunarLander-v2", render_mode="human"),
            actor_critic=core.MLPActorCritic,
            ac_kwargs=dict(hidden_sizes=[64, 64], activation=nn.Tanh),
            test_epochs=10,
            model_path="vpg.pt",
        )
    else:
        vpg(
            lambda: gym.make("LunarLander-v2"),
            actor_critic=core.MLPActorCritic,
            ac_kwargs=dict(hidden_sizes=[64, 64], activation=nn.Tanh),
            epoch=500,
            max_steps=5000,
            max_ep_len=1000,
            train_v_iters=80,
            pi_lr=3e-4,
            vf_lr=1e-3,
            gamma=0.99,
            lam=0.97,
        )
