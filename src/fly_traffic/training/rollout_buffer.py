"""Recurrent rollout buffer with sequence-chunked minibatch sampling for PPO."""

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import torch


@dataclass
class SequenceMinibatch:
    """Mini-batch of fixed-length trajectory chunks with initial recurrent states."""

    obs_seq: torch.Tensor
    actions_seq: torch.Tensor
    old_log_probs_seq: torch.Tensor
    advantages_seq: torch.Tensor
    returns_seq: torch.Tensor
    old_values_seq: torch.Tensor
    episode_starts_seq: torch.Tensor
    decision_masks_seq: torch.Tensor
    initial_hidden: torch.Tensor


class RecurrentRolloutBuffer:
    """Stores rollout transitions and hidden states for sequence-chunked PPO updates."""

    def __init__(
        self,
        rollout_steps: int,
        num_envs: int,
        obs_dim: int,
        hidden_dim: int,
        device: torch.device | str = "cpu",
    ) -> None:
        self.rollout_steps = rollout_steps
        self.num_envs = num_envs
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.device = torch.device(device)

        self.observations = torch.zeros(
            rollout_steps,
            num_envs,
            obs_dim,
            dtype=torch.float32,
            device=self.device,
        )
        self.actions = torch.zeros(
            rollout_steps,
            num_envs,
            dtype=torch.long,
            device=self.device,
        )
        self.log_probs = torch.zeros(
            rollout_steps,
            num_envs,
            dtype=torch.float32,
            device=self.device,
        )
        self.rewards = torch.zeros(
            rollout_steps,
            num_envs,
            dtype=torch.float32,
            device=self.device,
        )
        self.episode_starts = torch.zeros(
            rollout_steps,
            num_envs,
            dtype=torch.float32,
            device=self.device,
        )
        self.step_dones = torch.zeros(
            rollout_steps,
            num_envs,
            dtype=torch.float32,
            device=self.device,
        )
        self.decision_masks = torch.ones(
            rollout_steps,
            num_envs,
            dtype=torch.float32,
            device=self.device,
        )
        self.values = torch.zeros(
            rollout_steps,
            num_envs,
            dtype=torch.float32,
            device=self.device,
        )
        self.hidden_states = torch.zeros(
            rollout_steps,
            num_envs,
            hidden_dim,
            dtype=torch.float32,
            device=self.device,
        )

        self.advantages = torch.zeros(
            rollout_steps,
            num_envs,
            dtype=torch.float32,
            device=self.device,
        )
        self.returns = torch.zeros(
            rollout_steps,
            num_envs,
            dtype=torch.float32,
            device=self.device,
        )
        self._write_index = 0

    @property
    def obs(self) -> torch.Tensor:
        """Backward-compatible alias for rollout observation buffer."""
        return self.observations

    def reset(self) -> None:
        """Reset insertion cursor at the start of a new rollout collection phase."""
        self._write_index = 0

    def add(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        log_prob: torch.Tensor,
        reward: torch.Tensor,
        episode_start: torch.Tensor,
        step_done: torch.Tensor,
        value: torch.Tensor,
        hidden_before_step: torch.Tensor,
        decision_mask: torch.Tensor | None = None,
    ) -> None:
        """Store one vectorized rollout step across all parallel environments."""
        step_index = self._write_index
        if step_index >= self.rollout_steps:
            raise IndexError("RecurrentRolloutBuffer capacity exceeded.")

        self.observations[step_index].copy_(obs)
        self.actions[step_index].copy_(action)
        self.log_probs[step_index].copy_(log_prob)
        self.rewards[step_index].copy_(reward)
        self.episode_starts[step_index].copy_(episode_start)
        self.step_dones[step_index].copy_(step_done)

        if decision_mask is not None:
            self.decision_masks[step_index].copy_(decision_mask)
        else:
            self.decision_masks[step_index].fill_(1.0)

        self.values[step_index].copy_(value)
        self.hidden_states[step_index].copy_(hidden_before_step)
        self._write_index += 1

    def compute_advantages(
        self,
        last_value: torch.Tensor,
        last_done: torch.Tensor,
        gamma: float,
        gae_lambda: float,
    ) -> None:
        """Compute Generalized Advantage Estimation (GAE-Lambda) returns."""
        if self._write_index != self.rollout_steps:
            raise RuntimeError(
                f"Buffer incomplete: expected {self.rollout_steps} steps, "
                f"got {self._write_index}."
            )

        running_gae_advantage = torch.zeros(
            self.num_envs,
            dtype=torch.float32,
            device=self.device,
        )

        # Accumulate discounted TD residuals backwards from horizon end.
        for step_index in reversed(range(self.rollout_steps)):
            if step_index == self.rollout_steps - 1:
                next_non_terminal = 1.0 - last_done
                next_state_value = last_value
            else:
                next_non_terminal = 1.0 - self.step_dones[step_index]
                next_state_value = self.values[step_index + 1]

            temporal_difference_error = (
                self.rewards[step_index]
                + gamma * next_state_value * next_non_terminal
                - self.values[step_index]
            )
            running_gae_advantage = (
                temporal_difference_error
                + gamma * gae_lambda * next_non_terminal * running_gae_advantage
            )
            self.advantages[step_index] = running_gae_advantage

        self.returns = self.advantages + self.values

    def iter_sequence_minibatches(
        self,
        chunk_len: int,
        num_minibatches: int,
        rng: np.random.Generator | None = None,
    ) -> Iterator[SequenceMinibatch]:
        """Yield shuffled fixed-length sequence chunks with exact initial hidden states."""
        if self.rollout_steps % chunk_len != 0:
            raise ValueError(
                f"rollout_steps ({self.rollout_steps}) must be divisible by "
                f"chunk_len ({chunk_len})."
            )

        chunks_per_environment = self.rollout_steps // chunk_len
        total_chunks = chunks_per_environment * self.num_envs

        if num_minibatches <= 0 or total_chunks % num_minibatches != 0:
            raise ValueError(
                f"total_chunks ({total_chunks}) must be divisible by "
                f"num_minibatches ({num_minibatches})."
            )

        def _reshape_into_sequence_chunks(tensor: torch.Tensor) -> torch.Tensor:
            trailing_shape = tensor.shape[2:]
            chunked = tensor.reshape(
                chunks_per_environment,
                chunk_len,
                self.num_envs,
                *trailing_shape,
            )
            # Reorder [chunk_block, step, env, ...] -> [total_chunks, chunk_len, ...]
            return chunked.permute(0, 2, 1, *range(3, chunked.ndim)).reshape(
                total_chunks,
                chunk_len,
                *trailing_shape,
            )

        observation_chunks = _reshape_into_sequence_chunks(self.observations)
        action_chunks = _reshape_into_sequence_chunks(self.actions)
        log_prob_chunks = _reshape_into_sequence_chunks(self.log_probs)
        advantage_chunks = _reshape_into_sequence_chunks(self.advantages)
        return_chunks = _reshape_into_sequence_chunks(self.returns)
        value_chunks = _reshape_into_sequence_chunks(self.values)
        episode_start_chunks = _reshape_into_sequence_chunks(self.episode_starts)
        decision_mask_chunks = _reshape_into_sequence_chunks(self.decision_masks)

        # Slice the exact initial hidden state recorded at step k * chunk_len.
        initial_chunk_hidden_states = self.hidden_states[::chunk_len].reshape(
            total_chunks,
            self.hidden_dim,
        )

        generator = rng or np.random.default_rng()
        shuffled_chunk_indices = generator.permutation(total_chunks)
        chunks_per_minibatch = total_chunks // num_minibatches

        for minibatch_index in range(num_minibatches):
            start_offset = minibatch_index * chunks_per_minibatch
            end_offset = (minibatch_index + 1) * chunks_per_minibatch
            selected_chunk_indices = torch.as_tensor(
                shuffled_chunk_indices[start_offset:end_offset],
                dtype=torch.long,
                device=self.device,
            )

            yield SequenceMinibatch(
                obs_seq=observation_chunks[selected_chunk_indices].transpose(0, 1),
                actions_seq=action_chunks[selected_chunk_indices].transpose(0, 1),
                old_log_probs_seq=log_prob_chunks[selected_chunk_indices].transpose(0, 1),
                advantages_seq=advantage_chunks[selected_chunk_indices].transpose(0, 1),
                returns_seq=return_chunks[selected_chunk_indices].transpose(0, 1),
                old_values_seq=value_chunks[selected_chunk_indices].transpose(0, 1),
                episode_starts_seq=episode_start_chunks[selected_chunk_indices].transpose(0, 1),
                decision_masks_seq=decision_mask_chunks[selected_chunk_indices].transpose(0, 1),
                initial_hidden=initial_chunk_hidden_states[selected_chunk_indices],
            )
