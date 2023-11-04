import torch
import gymnasium as gym
import core
import numpy as np
import torch.nn as nn
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)


def ppo(
    env_fn,
    ac_constructor,
    ac_kwargs,
    epochs,
    max_steps,
    max_ep_len,
    gamma,
    lam,
    clip_ratio,
    pi_lr,
    vf_lr,
    train_pi_iters,
    train_v_iters,
    target_kl,
    logger,
    save_path,
):
    env = env_fn()
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape
    logger.info(f"Observation space: {obs_dim}, Action space: {act_dim}")
    buf = core.GAEBuffer(obs_dim, act_dim, max_steps, gamma, lam)
    ac = ac_constructor(env.observation_space, env.action_space, **ac_kwargs)

    pi_optimizer = torch.optim.Adam(ac.pi.parameters(), lr=pi_lr)
    vf_optimizer = torch.optim.Adam(ac.v.parameters(), lr=vf_lr)

    def compute_loss_pi(data):
        obs, act, adv, logp_old = data["obs"], data["act"], data["adv"], data["logp"]

        # policy loss
        pi, logp = ac.pi(obs, act)
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * adv
        loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()

        # extra useful info
        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item()
        clipped = ratio.gt(1 + clip_ratio) | ratio.lt(1 - clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)
        return loss_pi, pi_info

    def compute_loss_v(data):
        obs, ret = data["obs"], data["ret"]
        return ((ac.v(obs) - ret) ** 2).mean()

    def update():
        data = buf.get()

        pi_l_old, pi_info_old = compute_loss_pi(data)
        pi_l_old = pi_l_old.item()
        v_l_old = compute_loss_v(data).item()

        for i in range(train_pi_iters):
            pi_optimizer.zero_grad()
            loss_pi, pi_info = compute_loss_pi(data)
            loss_pi.backward()
            pi_optimizer.step()

            if pi_info["kl"] > 1.5 * target_kl:
                logger.info("Early stopping at step %d due to reaching max kl." % i)
                break

        for i in range(train_v_iters):
            vf_optimizer.zero_grad()
            loss_v = compute_loss_v(data)
            loss_v.backward()
            vf_optimizer.step()

        return loss_pi.item(), loss_v.item()

    o, ep_ret, ep_len = env.reset()[0], 0, 0
    best_avg_ret = -np.inf
    for epoch in range(epochs):
        ep_avg_ret, n_eps = 0, 0
        for step in range(max_steps):
            a, v, logp = ac.step(torch.as_tensor(o, dtype=torch.float32))
            next_o, r, d, t, _ = env.step(a)

            buf.store(o, a, v, r, logp)
            ep_ret += r
            ep_len += 1
            ep_avg_ret += r

            o = next_o

            timeout = ep_len == max_ep_len
            terminal = d or t or timeout
            epoch_ended = step == max_steps - 1

            if terminal or epoch_ended:
                if epoch_ended and not(terminal):
                    print(f"Trajectory cut off by epoch at {ep_len} steps.", flush=True)
                if timeout or epoch_ended:
                    _, v, _ = ac.step(torch.as_tensor(o, dtype=torch.float32))
                else:
                    v = 0
                buf.finish_path(v)
                n_eps += 1
                o, ep_ret, ep_len = env.reset()[0], 0, 0

        loss_pi, loss_v = update()

        ep_avg_ret = ep_avg_ret / n_eps
        if ep_avg_ret > best_avg_ret:
            best_avg_ret = ep_avg_ret
            torch.save(ac.state_dict(), save_path)
        logger.info(
            f"Epoch: {epoch}, Loss pi: {loss_pi}, Loss v: {loss_v}, Rew: {ep_avg_ret}"
        )


def ppo_test(env_fn, actor_critic, ac_kwargs, test_epochs, model_path):
    env = env_fn()
    ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs)
    ac.load_state_dict(torch.load(model_path))
    ac.eval()

    for epoch in range(test_epochs):
        o, ep_ret, ep_len = env.reset()[0], 0, 0
        while True:
            a, v, logp = ac.step(torch.as_tensor(o, dtype=torch.float32))
            next_o, r, d, tructated, _ = env.step(a)
            ep_ret += r
            ep_len += 1
            o = next_o
            terminal = d or tructated
            if terminal:
                print(f"Epoch: {epoch}, Reward: {ep_ret}")
                o, ep_ret, ep_len = env.reset()[0], 0, 0
                break
        cmd = input("Continue? [y/n]")
        if cmd == "n":
            break


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()
    env_name = "LunarLander-v2"
    env_name = "CartPole-v1"
    if args.test:
        ppo_test(
            lambda: gym.make(env_name, render_mode="human"),
            actor_critic=core.MLPActorCritic,
            ac_kwargs=dict(hidden_sizes=[64, 64], activation=nn.Tanh),
            test_epochs=10,
            model_path="ppo.pt",
        )
    else:
        ppo(
            lambda: gym.make(env_name),
            ac_constructor=core.MLPActorCritic,
            ac_kwargs=dict(hidden_sizes=[64, 64], activation=nn.Tanh),
            epochs=500,
            max_steps=5000,
            max_ep_len=1000,
            gamma=0.99,
            lam=0.97,
            clip_ratio=0.2,
            pi_lr=3e-4,
            vf_lr=1e-3,
            train_pi_iters=80,
            train_v_iters=80,
            target_kl=0.01,
            logger=logging.getLogger(__name__),
            save_path="ppo.pt",
        )
