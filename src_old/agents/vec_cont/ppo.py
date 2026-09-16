"""PPO agent for engineered-vector, continuous-action CarRacing."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal
from torch.nn import functional as F

from src.agents.vec_cont.rollout_buffer import RolloutBatch


class VectorActorCritic(nn.Module):
    """Gaussian policy and value function for a vector state."""

    def __init__(
        self, state_dim: int, action_dim: int, *, initial_log_std: float = 0.0
    ) -> None:
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 256), nn.Tanh(), nn.Linear(256, 256), nn.Tanh()
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 256), nn.Tanh(), nn.Linear(256, 256), nn.Tanh()
        )
        self.action_mean = nn.Linear(256, action_dim)
        self.value_head = nn.Linear(256, 1)
        self.log_std = nn.Parameter(torch.full((action_dim,), initial_log_std))
        self.apply(self._initialize)
        nn.init.orthogonal_(self.action_mean.weight, gain=0.01)
        nn.init.zeros_(self.action_mean.bias)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.zeros_(self.value_head.bias)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2.0))
            nn.init.zeros_(module.bias)

    def forward(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.action_mean(self.actor(states)), self.value_head(self.critic(states)).squeeze(1)


class PPOAgent:
    """Clipped PPO with a tanh-squashed Gaussian CarRacing policy."""

    def __init__(
        self,
        state_dim: int,
        action_space,
        device: torch.device,
        *,
        learning_rate: float = 3e-4,
        clip_ratio: float = 0.2,
        value_coefficient: float = 0.5,
        entropy_coefficient: float = 0.0,
        max_grad_norm: float = 0.5,
        initial_log_std: float = 0.0,
    ) -> None:
        self.device = device
        self.action_dim = int(np.prod(action_space.shape))
        self.clip_ratio = clip_ratio
        self.value_coefficient = value_coefficient
        self.entropy_coefficient = entropy_coefficient
        self.max_grad_norm = max_grad_norm
        self.actor_critic = VectorActorCritic(
            state_dim, self.action_dim, initial_log_std=initial_log_std
        ).to(device)
        self.optimizer = torch.optim.Adam(self.actor_critic.parameters(), lr=learning_rate, eps=1e-5)
        low = torch.as_tensor(action_space.low, dtype=torch.float32, device=device)
        high = torch.as_tensor(action_space.high, dtype=torch.float32, device=device)
        self.action_scale = (high - low) / 2.0
        self.action_bias = (high + low) / 2.0
        self.optimization_steps = 0

    def set_learning_rate(self, learning_rate: float) -> None:
        """Set the optimizer learning rate for a training schedule."""
        for group in self.optimizer.param_groups:
            group["lr"] = learning_rate

    def set_entropy_coefficient(self, entropy_coefficient: float) -> None:
        """Set the entropy bonus weight for a training schedule."""
        self.entropy_coefficient = entropy_coefficient

    def _distribution(self, states: torch.Tensor) -> tuple[Normal, torch.Tensor]:
        mean, values = self.actor_critic(states)
        return Normal(mean, self.actor_critic.log_std.exp().expand_as(mean)), values

    def _log_prob(self, distribution: Normal, actions: torch.Tensor) -> torch.Tensor:
        normalized = ((actions - self.action_bias) / self.action_scale).clamp(-0.999999, 0.999999)
        latent = torch.atanh(normalized)
        correction = torch.log(self.action_scale * (1.0 - normalized.square()) + 1e-6)
        return (distribution.log_prob(latent) - correction).sum(dim=1)

    @torch.no_grad()
    def select_action(
        self, state: np.ndarray, *, deterministic: bool = False
    ) -> tuple[np.ndarray, float, float]:
        states = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        distribution, values = self._distribution(states)
        latent = distribution.mean if deterministic else distribution.sample()
        normalized = torch.tanh(latent)
        actions = normalized * self.action_scale + self.action_bias
        log_prob = self._log_prob(distribution, actions)
        return (
            actions.squeeze(0).cpu().numpy().astype(np.float32),
            float(log_prob.item()),
            float(values.item()),
        )

    @torch.no_grad()
    def value(self, state: np.ndarray) -> float:
        states = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        _, values = self._distribution(states)
        return float(values.item())

    def update(self, batch: RolloutBatch, *, epochs: int, minibatch_size: int) -> dict[str, float]:
        advantages = (batch.advantages - batch.advantages.mean()) / (batch.advantages.std() + 1e-8)
        batch_size = batch.states.shape[0]
        metrics: dict[str, list[float]] = {"policy_loss": [], "value_loss": [], "entropy": [], "approx_kl": []}

        for _ in range(epochs):
            indices = torch.randperm(batch_size, device=self.device)
            for start in range(0, batch_size, minibatch_size):
                index = indices[start : start + minibatch_size]
                distribution, values = self._distribution(batch.states[index])
                log_probs = self._log_prob(distribution, batch.actions[index])
                log_ratio = log_probs - batch.old_log_probs[index]
                ratio = log_ratio.exp()
                surrogate_a = ratio * advantages[index]
                surrogate_b = ratio.clamp(1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * advantages[index]
                policy_loss = -torch.minimum(surrogate_a, surrogate_b).mean()
                value_loss = F.mse_loss(values, batch.returns[index])
                entropy = distribution.entropy().sum(dim=1).mean()
                loss = policy_loss + self.value_coefficient * value_loss - self.entropy_coefficient * entropy

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()
                self.optimization_steps += 1

                metrics["policy_loss"].append(float(policy_loss.item()))
                metrics["value_loss"].append(float(value_loss.item()))
                metrics["entropy"].append(float(entropy.item()))
                metrics["approx_kl"].append(float((ratio - 1.0 - log_ratio).mean().item()))

        return {name: float(np.mean(values)) for name, values in metrics.items()}

    def save(self, path: str | Path, **metadata: Any) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "action_dim": self.action_dim,
                "actor_critic": self.actor_critic.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "optimization_steps": self.optimization_steps,
                "metadata": metadata,
            },
            path,
        )

    def load(self, path: str | Path, *, load_optimizer: bool = True) -> dict[str, Any]:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        if checkpoint["action_dim"] != self.action_dim:
            raise ValueError("Checkpoint action_dim does not match current agent")
        self.actor_critic.load_state_dict(checkpoint["actor_critic"])
        if load_optimizer:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.optimization_steps = int(checkpoint.get("optimization_steps", 0))
        return dict(checkpoint.get("metadata", {}))
