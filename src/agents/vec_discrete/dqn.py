"""DQN network and optimization logic."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from src.agents.vec_discrete.replay_buffer import ReplayBatch


class QNetwork(nn.Module):
    """MLP that maps a vector state to one Q-value per discrete action."""

    def __init__(self, state_dim: int, action_dim: int) -> None:
        super().__init__()
        if state_dim <= 0 or action_dim <= 0:
            raise ValueError("state_dim and action_dim must be positive")

        self.network = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
        )

        self.apply(self._initialize_layer)

    @staticmethod
    def _initialize_layer(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2.0))
            nn.init.zeros_(module.bias)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)


class DQNAgent:
    """Online/target networks, epsilon-greedy action, and a Double-DQN update."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        device: torch.device,
        *,
        gamma: float = 0.99,
        learning_rate: float = 1e-4,
        max_grad_norm: float = 10.0,
    ) -> None:
        if not 0.0 <= gamma <= 1.0:
            raise ValueError("gamma must be in [0, 1]")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive")

        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.device = device
        self.gamma = float(gamma)
        self.max_grad_norm = float(max_grad_norm)

        self.online_network = QNetwork(state_dim, action_dim).to(device)
        self.target_network = QNetwork(state_dim, action_dim).to(device)
        self.target_network.load_state_dict(self.online_network.state_dict())
        self.target_network.eval()

        self.optimizer = torch.optim.AdamW(
            self.online_network.parameters(),
            lr=learning_rate,
            amsgrad=True,
        )

        self.optimization_steps = 0

    @torch.no_grad()
    def select_action(
        self,
        state: np.ndarray,
        epsilon: float,
        rng: np.random.Generator,
    ) -> int:
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon must be in [0, 1]")

        if rng.random() < epsilon:
            return int(rng.integers(self.action_dim))

        state_tensor = torch.as_tensor(
            state,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)
        q_values = self.online_network(state_tensor)
        return int(q_values.argmax(dim=1).item())

    def update(self, batch: ReplayBatch) -> dict[str, float]:
        """Perform one Double-DQN update using a frozen target network."""
        selected_q_values = self.online_network(batch.states).gather(
            dim=1,
            index=batch.actions.unsqueeze(1),
        ).squeeze(1)

        with torch.no_grad():
            # Online network chooses the next action; target network evaluates it.
            next_actions = self.online_network(batch.next_states).argmax(
                dim=1,
                keepdim=True,
            )
            next_q_values = self.target_network(batch.next_states).gather(
                dim=1,
                index=next_actions,
            ).squeeze(1)
            td_targets = batch.rewards + self.gamma * (
                1.0 - batch.terminated
            ) * next_q_values

        loss = F.smooth_l1_loss(selected_q_values, td_targets)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = nn.utils.clip_grad_norm_(
            self.online_network.parameters(), self.max_grad_norm
        )
        self.optimizer.step()

        self.optimization_steps += 1

        with torch.no_grad():
            td_error = td_targets - selected_q_values

        return {
            "loss": float(loss.item()),
            "mean_q": float(selected_q_values.mean().item()),
            "mean_target": float(td_targets.mean().item()),
            "mean_abs_td_error": float(td_error.abs().mean().item()),
            "gradient_norm": float(gradient_norm.item()),
        }

    def hard_update_target(self) -> None:
        self.target_network.load_state_dict(self.online_network.state_dict())

    def save(self, path: str | Path, **metadata: Any) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(
            {
                "state_dim": self.state_dim,
                "action_dim": self.action_dim,
                "online_network": self.online_network.state_dict(),
                "target_network": self.target_network.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "optimization_steps": self.optimization_steps,
                "metadata": metadata,
            },
            path,
        )

    def load(self, path: str | Path, *, load_optimizer: bool = True) -> dict[str, Any]:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        if checkpoint["state_dim"] != self.state_dim:
            raise ValueError("Checkpoint state_dim does not match current agent")
        if checkpoint["action_dim"] != self.action_dim:
            raise ValueError("Checkpoint action_dim does not match current agent")

        self.online_network.load_state_dict(checkpoint["online_network"])
        self.target_network.load_state_dict(checkpoint["target_network"])
        if load_optimizer:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.optimization_steps = int(checkpoint.get("optimization_steps", 0))

        return dict(checkpoint.get("metadata", {}))
