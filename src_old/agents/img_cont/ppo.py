"""PPO agent for stacked-image, continuous-action CarRacing."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal
from torch.nn import functional as F

from src.agents.img_cont.rollout_buffer import RolloutBatch


class ImageActorCritic(nn.Module):
    """CNN Gaussian policy and value function for stacked RGB observations."""

    def __init__(
        self,
        action_dim: int,
        observation_shape: tuple[int, ...],
        *,
        initial_log_std: float = 0.0,
        initial_action_mean_bias: tuple[float, ...] | None = None,
    ) -> None:
        super().__init__()
        num_frames, height, width, channels = observation_shape
        if channels != 3:
            raise ValueError("Expected stacked RGB observations")
        self.features = nn.Sequential(
            nn.Conv2d(num_frames * channels, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(), nn.Flatten(),
        )
        with torch.no_grad():
            feature_dim = int(self.features(torch.zeros(1, num_frames * channels, height, width)).shape[1])
        self.actor = nn.Sequential(nn.Linear(feature_dim, 512), nn.ReLU(), nn.Linear(512, action_dim))
        self.critic = nn.Sequential(nn.Linear(feature_dim, 512), nn.ReLU(), nn.Linear(512, 1))
        self.log_std = nn.Parameter(torch.full((action_dim,), initial_log_std))
        self.apply(self._initialize)
        nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)
        nn.init.zeros_(self.actor[-1].bias)
        if initial_action_mean_bias is not None:
            if len(initial_action_mean_bias) != action_dim:
                raise ValueError("initial_action_mean_bias has the wrong length")
            with torch.no_grad():
                self.actor[-1].bias.copy_(
                    torch.as_tensor(initial_action_mean_bias, dtype=torch.float32)
                )
        nn.init.orthogonal_(self.critic[-1].weight, gain=1.0)
        nn.init.zeros_(self.critic[-1].bias)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2.0))
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_frames, height, width, channels = images.shape
        images = images.permute(0, 1, 4, 2, 3).reshape(
            batch_size, num_frames * channels, height, width
        ).float().div_(255.0)
        features = self.features(images)
        return self.actor(features), self.critic(features).squeeze(1)


class PPOAgent:
    """Clipped PPO with a tanh-squashed Gaussian policy over RGB frames."""

    def __init__(
        self,
        action_space,
        observation_shape: tuple[int, ...],
        device: torch.device,
        *,
        learning_rate: float = 2.5e-4,
        clip_ratio: float = 0.2,
        value_coefficient: float = 0.5,
        entropy_coefficient: float = 0.0,
        max_grad_norm: float = 0.5,
        initial_log_std: float = 0.0,
        initial_action_mean_bias: tuple[float, ...] | None = None,
        target_kl: float | None = None,
    ) -> None:
        self.device = device
        self.action_dim = int(np.prod(action_space.shape))
        self.observation_shape = observation_shape
        self.clip_ratio = clip_ratio
        self.value_coefficient = value_coefficient
        self.entropy_coefficient = entropy_coefficient
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl
        self.actor_critic = ImageActorCritic(
            self.action_dim,
            observation_shape,
            initial_log_std=initial_log_std,
            initial_action_mean_bias=initial_action_mean_bias,
        ).to(device)
        self.optimizer = torch.optim.Adam(self.actor_critic.parameters(), lr=learning_rate, eps=1e-5)
        low = torch.as_tensor(action_space.low, dtype=torch.float32, device=device)
        high = torch.as_tensor(action_space.high, dtype=torch.float32, device=device)
        self.action_scale = (high - low) / 2.0
        self.action_bias = (high + low) / 2.0
        self.optimization_steps = 0

    def set_learning_rate(self, learning_rate: float) -> None:
        for group in self.optimizer.param_groups:
            group["lr"] = learning_rate

    def set_entropy_coefficient(self, entropy_coefficient: float) -> None:
        self.entropy_coefficient = entropy_coefficient

    def _distribution(self, images: torch.Tensor) -> tuple[Normal, torch.Tensor]:
        mean, values = self.actor_critic(images)
        log_std = self.actor_critic.log_std.clamp(-3.0, 1.0)
        return Normal(mean, log_std.exp().expand_as(mean)), values

    def _log_prob(self, distribution: Normal, actions: torch.Tensor) -> torch.Tensor:
        normalized = ((actions - self.action_bias) / self.action_scale).clamp(-0.999999, 0.999999)
        latent = torch.atanh(normalized)
        correction = torch.log(self.action_scale * (1.0 - normalized.square()) + 1e-6)
        return (distribution.log_prob(latent) - correction).sum(dim=1)

    @torch.no_grad()
    def select_action(self, image: np.ndarray, *, deterministic: bool = False) -> tuple[np.ndarray, float, float]:
        images = torch.as_tensor(image, device=self.device).unsqueeze(0)
        distribution, values = self._distribution(images)
        latent = distribution.mean if deterministic else distribution.sample()
        actions = torch.tanh(latent) * self.action_scale + self.action_bias
        log_prob = self._log_prob(distribution, actions)
        return actions.squeeze(0).cpu().numpy().astype(np.float32), float(log_prob.item()), float(values.item())

    @torch.no_grad()
    def value(self, image: np.ndarray) -> float:
        images = torch.as_tensor(image, device=self.device).unsqueeze(0)
        _, values = self._distribution(images)
        return float(values.item())

    def update(self, batch: RolloutBatch, *, epochs: int, minibatch_size: int) -> dict[str, float]:
        advantages = (batch.advantages - batch.advantages.mean()) / (batch.advantages.std() + 1e-8)
        batch_size = batch.images.shape[0]
        metrics: dict[str, list[float]] = {"policy_loss": [], "value_loss": [], "entropy": [], "approx_kl": []}
        stop_early = False
        for _ in range(epochs):
            indices = torch.randperm(batch_size, device=self.device)
            for start in range(0, batch_size, minibatch_size):
                index = indices[start : start + minibatch_size]
                distribution, values = self._distribution(batch.images[index])
                log_probs = self._log_prob(distribution, batch.actions[index])
                log_ratio = log_probs - batch.old_log_probs[index]
                ratio = log_ratio.exp()
                policy_loss = -torch.minimum(
                    ratio * advantages[index],
                    ratio.clamp(1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * advantages[index],
                ).mean()
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
                approx_kl = float((ratio - 1.0 - log_ratio).mean().item())
                metrics["approx_kl"].append(approx_kl)
                if self.target_kl is not None and approx_kl > 1.5 * self.target_kl:
                    stop_early = True
                    break
            if stop_early:
                break
        return {name: float(np.mean(values)) for name, values in metrics.items()}

    def save(self, path: str | Path, **metadata: Any) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"action_dim": self.action_dim, "observation_shape": self.observation_shape, "actor_critic": self.actor_critic.state_dict(), "optimizer": self.optimizer.state_dict(), "optimization_steps": self.optimization_steps, "metadata": metadata}, path)

    def load(self, path: str | Path, *, load_optimizer: bool = True) -> dict[str, Any]:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        if checkpoint["action_dim"] != self.action_dim:
            raise ValueError("Checkpoint action_dim does not match current agent")
        if tuple(checkpoint.get("observation_shape", ())) != self.observation_shape:
            raise ValueError("Checkpoint observation shape does not match current agent")
        self.actor_critic.load_state_dict(checkpoint["actor_critic"])
        if load_optimizer:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.optimization_steps = int(checkpoint.get("optimization_steps", 0))
        return dict(checkpoint.get("metadata", {}))
