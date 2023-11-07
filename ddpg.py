import torch
import numpy as np
import core
from copy import deepcopy
import gymnasium as gym


class ReplayBuffer(object):
    def __init__(self, obs_dim, act_dim, size) -> None:
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(
            obs=self.obs_buf[idxs],
            obs2=self.obs2_buf[idxs],
            act=self.act_buf[idxs],
            rew=self.rew_buf[idxs],
            done=self.done_buf[idxs],
        )
        return {k: torch.as_tensor(v) for k, v in batch.items()}


def ddpg(
    env_fn,
    ac_constructor,
    ac_kwargs,
    epochs,
    steps_per_epoch,
    replay_size=int(1e6),
    gamma=0.99,
    polyak=0.995,
    pi_lr=1e-3,
    q_lr=1e-3,
    batch_size=100,
    start_steps=10000,
    update_after=1000,
    update_every=50,
    act_noise=0.1,
    num_test_episodes=10,
    max_ep_len=1000,
):
    env, test_env = env_fn(), env_fn()
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape[0]

    act_limit = env.action_shape.high[0]
    ac = ac_constructor(env.observation_space, env.action_space, **ac_kwargs)
    ac_targ = deepcopy(ac)

    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)

    for p in ac_targ.parameters():
        p.requres_grad = False

    def compute_loss_q(data):
        o, a, r, o2, d = (
            data["obs"],
            data["act"],
            data["rew"],
            data["obs2"],
            data["done"],
        )
        q = ac.q(o, a)

        with torch.no_grad():
            q_pi_targ = ac_targ.q(o2, ac_targ.pi(o2))
            backup = r + gamma * (1 - d) * q_pi_targ

        loss_q = ((q - backup) ** 2).mean()
        loss_info = dict(QVals=q.detach().numpy())
        return loss_q, loss_info

    def compute_loss_pi(data):
        o = data["obs"]
        q_pi = ac.q(o, ac.pi(o))
        return -q_pi.mean()

    pi_optimizer = torch.optim.Adam(ac.pi.parameters(), lr=pi_lr)
    q_optimizer = torch.optim.Adam(ac.q.parameters(), lr=q_lr)

    def update(data):
        q_optimizer.zero_grad()
        loss_q, loss_info = compute_loss_q(data)
        loss_q.backward()
        q_optimizer.step()

        for p in ac.q.parameters():
            p.requires_grad = False

        pi_optimizer.zero_grad()
        loss_pi = compute_loss_pi(data)
        loss_pi.backward()
        pi_optimizer.step()

        for p in ac.q.parameters():
            p.requires_grad = True

        with torch.no_grad():
            for p, p_targ in zip(ac.parameters(), ac_targ.parameters()):
                p_targ.data.mul_(polyak)
                p_targ.data.add_((1 - polyak) * p.data)

    def get_action(o, noise_scale):
        a = ac.act(torch.as_tensor(o, dtype=torch.float32))
        a += noise_scale * np.random.randn(act_dim)
        return np.clip(a, -act_limit, act_limit)

    def test_agent():
        for _ in range(num_test_episodes):
            o, d, ep_ret, ep_len = test_env.reset()[0], False, 0, 0
            while not (d or (ep_len == max_ep_len)):
                o, r, terminal, trunctated, _ = test_env.step(get_action(o, 0))
                d = terminal or trunctated
            ep_ret += r
            ep_len += 1
        return ep_ret, ep_len

    total_steps = steps_per_epoch * epochs
    o, ep_ret, ep_len = env.reset()[0], 0, 0
    best_avg_ret = -np.inf

    for t in range(total_steps):
        if t > start_steps:
            a = get_action(o, act_noise)
        else:
            a = env.action_space.sample()

        (
            o2,
            r,
            terminal,
            trunctated,
            _,
        ) = env.step(a)
        d = terminal or trunctated
        ep_ret += r
        ep_len += 1

        d = False if ep_len == max_ep_len else d

        replay_buffer.store(o, a, r, o2, d)

        o = o2
        if d or (ep_len == max_ep_len):
            o, ep_ret, ep_len = env.reset()[0], 0, 0

        if t >= update_after and t % update_every == 0:
            for _ in range(update_every):
                batch = replay_buffer.sample_batch(batch_size)
                update(data=batch)

        if (t + 1) % steps_per_epoch == 0:
            epoch = (t + 1) // steps_per_epoch
            ret, len = test_agent()
            avg_ep_ret = ret / len
            if avg_ep_ret > best_avg_ret:
                best_avg_ret = avg_ep_ret
                torch.save(ac.state_dict(), "ddpg.pt")
            print(f"Epoch: {epoch}, Average ep ret: {avg_ep_ret}")


if __name__ == "__main__":
    ddpg(
        lambda: gym.make("CarPole-v1"),
        ac_constructor=core.MLPQActorCritic,
        ac_kwargs=dict(hidden_sizes=[256, 256]),
        epochs=50,
    )
