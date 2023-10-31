import torch
import gymnasium as gym
from typing import List
import cv2
import numpy as np


def get_policy_network(
    sizes: List[int],
    activation: torch.nn.Module = torch.nn.ReLU,
    output_activation: torch.nn.Module = torch.nn.Identity,
):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [torch.nn.Linear(sizes[j], sizes[j + 1]), act()]
    return torch.nn.Sequential(*layers)


def get_policy(obs: torch.Tensor, policy_network: torch.nn.Module):
    logits = policy_network(obs)
    return torch.distributions.Categorical(logits=logits)


def sample_action(obs: torch.Tensor, policy_network: torch.nn.Module):
    return get_policy(obs, policy_network).sample().item()


def compute_loss(
    obs: torch.Tensor,
    act: torch.Tensor,
    weights: torch.Tensor,
    policy_network: torch.nn.Module,
):
    logp = get_policy(obs, policy_network).log_prob(act)
    return -(logp * weights).mean()


env = gym.make("CartPole-v1", render_mode="rgb_array")
obs_dim = env.observation_space.shape[0]
n_acts = env.action_space.n
policy_network = get_policy_network(sizes=[obs_dim, 64, 32, n_acts])
policy_network.train()
optimizer = torch.optim.Adam(policy_network.parameters(), lr=1e-2)
batch_size = 5000
epochs = 100


def train_one_epoch():
    batch_obs = []
    batch_acts = []
    batch_weights = []
    batch_rewards = []
    batch_lens = []

    ep_rewards = []
    obs = env.reset()[0]

    finish_rendering_this_epoch = False

    while True:
        batch_obs.append(obs.copy())
        act = sample_action(torch.as_tensor(obs, dtype=torch.float32), policy_network)
        if not finish_rendering_this_epoch:
            frame = env.render()
            cv2.imshow("frame", frame)
            cv2.waitKey(1)
            
        obs, rew, terminated, truncated, _ = env.step(act)

        batch_acts.append(act)
        ep_rewards.append(rew)
        if terminated or truncated:
            ep_sum, ep_len = sum(ep_rewards), len(ep_rewards)
            batch_rewards.append(ep_sum)
            batch_lens.append(ep_len)
            batch_weights += [ep_sum] * ep_len

            finish_rendering_this_epoch = True
            obs, ep_rewards = env.reset()[0], []
            if len(batch_obs) > batch_size:
                break

    optimizer.zero_grad()
    batch_loss = compute_loss(
        obs=torch.as_tensor(np.array(batch_obs), dtype=torch.float32),
        act=torch.as_tensor(np.array(batch_acts), dtype=torch.int32),
        weights=torch.as_tensor(np.array(batch_weights), dtype=torch.float32),
        policy_network=policy_network,
    )
    batch_loss.backward()
    optimizer.step()
    return batch_loss, batch_rewards, batch_lens

best_reward = 0
for i in range(epochs):
    batch_loss, batch_rewards, batch_lens = train_one_epoch()
    print(
        f"epoch: {i:3d} loss: {batch_loss:.3f} reward: {np.mean(batch_rewards):.3f} ep_len: {np.mean(batch_lens):3f}"
    )
    if np.mean(batch_rewards) > best_reward:
        best_reward = np.mean(batch_rewards)
        torch.save(policy_network.state_dict(), "policy_network.pt")