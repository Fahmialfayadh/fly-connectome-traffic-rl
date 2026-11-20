"""Abstract interfaces and adapters for RL policies and rule-based traffic agents."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical


@dataclass
class PolicyStepOutput:
    """Output bundle returned by recurrent actor-critic policy queries."""

    action: torch.Tensor
    log_prob: torch.Tensor
    entropy: torch.Tensor
    value: torch.Tensor
    next_hidden: torch.Tensor


class BaseTrafficAgent(ABC):
    """Uniform agent interface shared by evaluators and the interactive Pygame UI."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable factual agent identifier."""

    @abstractmethod
    def reset(self) -> None:
        """Reset any internal recurrent state at the start of an episode."""

    @abstractmethod
    def select_action(self, obs: np.ndarray, deterministic: bool = True) -> int:
        """Choose a discrete phase action given the current observation vector."""


class BaseRecurrentActorCritic(nn.Module, ABC):
    """Base PyTorch actor-critic policy supporting single-step and sequence rollouts."""

    @property
    @abstractmethod
    def hidden_dim(self) -> int:
        """Dimensionality of the recurrent hidden state vector."""

    @abstractmethod
    def initial_hidden(
        self,
        batch_size: int,
        device: torch.device | str = "cpu",
    ) -> torch.Tensor:
        """Return zero-initialized hidden state tensor of shape [batch_size, hidden_dim]."""

    @abstractmethod
    def forward_step(
        self,
        obs: torch.Tensor,
        hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run one timestep and return (action_logits, value, next_hidden)."""

    def forward_sequence(
        self,
        obs_seq: torch.Tensor,
        initial_hidden: torch.Tensor,
        dones_seq: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Unroll policy across [seq_len, batch, obs_dim] resetting hidden on dones_seq."""
        seq_len = obs_seq.shape[0]
        hidden = initial_hidden
        logits_list: list[torch.Tensor] = []
        values_list: list[torch.Tensor] = []

        for step_idx in range(seq_len):
            reset_mask = (1.0 - dones_seq[step_idx].to(hidden.dtype)).unsqueeze(-1)
            hidden = hidden * reset_mask
            logits, value, hidden = self.forward_step(obs_seq[step_idx], hidden)
            logits_list.append(logits)
            values_list.append(value)

        return torch.stack(logits_list, dim=0), torch.stack(values_list, dim=0), hidden

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        hidden: torch.Tensor,
        action: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> PolicyStepOutput:
        """Evaluate action distribution, sample or score action, and estimate state value."""
        logits, value, next_hidden = self.forward_step(obs, hidden)
        dist = Categorical(logits=logits)

        if action is None:
            chosen_action = torch.argmax(logits, dim=-1) if deterministic else dist.sample()
        else:
            chosen_action = action

        return PolicyStepOutput(
            action=chosen_action,
            log_prob=dist.log_prob(chosen_action),
            entropy=dist.entropy(),
            value=value,
            next_hidden=next_hidden,
        )

    def count_trainable_parameters(self) -> int:
        """Return number of parameters with requires_grad=True."""
        return sum(param.numel() for param in self.parameters() if param.requires_grad)

    def count_total_parameters(self) -> int:
        """Return total number of parameter elements (excluding non-parameter buffers)."""
        return sum(param.numel() for param in self.parameters())


class NeuralAgentAdapter(BaseTrafficAgent):
    """Adapts any trained BaseRecurrentActorCritic into a stateful BaseTrafficAgent."""

    def __init__(
        self,
        policy: BaseRecurrentActorCritic,
        agent_name: str,
        device: torch.device | str = "cpu",
    ) -> None:
        self.policy = policy
        self._agent_name = agent_name
        self.device = torch.device(device)
        self.policy.to(self.device)
        self.policy.eval()
        self._hidden = self.policy.initial_hidden(batch_size=1, device=self.device)
        self._latest_action_probabilities: np.ndarray = np.full(4, 0.25, dtype=np.float32)
        self._latest_value_estimate: float = 0.0

    @property
    def name(self) -> str:
        """Human-readable agent label."""
        return self._agent_name

    @property
    def latest_hidden_numpy(self) -> np.ndarray:
        """Return most recent hidden activation vector on CPU as 1D float32 array."""
        return self._hidden.squeeze(0).detach().cpu().numpy().astype(np.float32)

    @property
    def latest_action_probabilities(self) -> np.ndarray:
        """Return softmax phase probabilities from the latest inference step."""
        return self._latest_action_probabilities.copy()

    @property
    def latest_value_estimate(self) -> float:
        """Return critic state-value estimate V(s_t) from the latest inference step."""
        return float(self._latest_value_estimate)

    def reset(self) -> None:
        """Zero out recurrent hidden state for a fresh episode."""
        self._hidden = self.policy.initial_hidden(batch_size=1, device=self.device)
        self._latest_action_probabilities = np.full(4, 0.25, dtype=np.float32)
        self._latest_value_estimate = 0.0

    def select_action(self, obs: np.ndarray, deterministic: bool = True) -> int:
        """Run one inference step without gradient tracking and advance hidden state."""
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action_logits, state_value, next_hidden = self.policy.forward_step(
                obs_tensor,
                self._hidden,
            )
            action_probs = torch.softmax(action_logits, dim=-1).squeeze(0)

            if deterministic:
                chosen_action = int(torch.argmax(action_logits, dim=-1).item())
            else:
                chosen_action = int(Categorical(logits=action_logits).sample().item())

        self._hidden = next_hidden
        self._latest_action_probabilities = action_probs.detach().cpu().numpy().astype(np.float32)
        self._latest_value_estimate = float(state_value.squeeze(0).item())
        return chosen_action
