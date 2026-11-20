"""Frozen sparse Drosophila connectome leaky recurrent cell with E/I modulation."""

import math

import numpy as np
import torch
from torch import nn

from fly_traffic.connectome.artifact import ConnectomeArtifact


class ConnectomeRecurrentCell(nn.Module):
    """Sparse biological recurrent core with frozen E/I wiring and live neuromodulation."""

    def __init__(
        self,
        obs_dim: int,
        artifact: ConnectomeArtifact,
        initial_leak_rate: float = 0.35,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.num_neurons = artifact.metadata.num_neurons

        w_exc_csr, w_inh_csr = artifact.to_sparse_csr_tensors(device="cpu")
        self.register_buffer("w_exc_csr", w_exc_csr, persistent=False)
        self.register_buffer("w_inh_csr", w_inh_csr, persistent=False)
        self.register_buffer(
            "lesion_mask",
            torch.ones(1, self.num_neurons, dtype=torch.float32),
            persistent=False,
        )

        self.input_projection = nn.Linear(obs_dim, self.num_neurons)
        nn.init.orthogonal_(self.input_projection.weight, gain=1.0)
        nn.init.zeros_(self.input_projection.bias)

        clamped_tau = min(max(initial_leak_rate, 0.06), 0.94)
        normalized = (clamped_tau - 0.05) / 0.90
        init_logit = math.log(normalized / (1.0 - normalized))
        self.tau_logit = nn.Parameter(torch.tensor(init_logit, dtype=torch.float32))

        self.g_exc: float = 1.0
        self.g_inh: float = 1.0

    @property
    def leak_rate(self) -> torch.Tensor:
        """Bounded neural integration leak rate tau in (0.05, 0.95)."""
        return 0.05 + 0.90 * torch.sigmoid(self.tau_logit)

    def set_pharmacology(self, g_exc: float, g_inh: float) -> None:
        """Set excitatory (ACh) and inhibitory (GABA/Glu) synaptic gain multipliers."""
        self.g_exc = float(max(0.0, g_exc))
        self.g_inh = float(max(0.0, g_inh))

    def reset_lesion(self) -> None:
        """Restore all neurons to active (unlesioned) status."""
        self.lesion_mask.fill_(1.0)

    def active_neuron_count(self) -> int:
        """Return count of unlesioned neurons currently active."""
        return int((self.lesion_mask > 0.5).sum().item())

    def apply_random_lesion(
        self,
        fraction: float,
        seed: int | None = None,
    ) -> int:
        """Silence a fraction of all neurons uniformly at random; return silenced count."""
        self.reset_lesion()
        clamped = min(max(fraction, 0.0), 1.0)
        count = int(round(clamped * self.num_neurons))
        if count <= 0:
            return 0

        rng = np.random.default_rng(seed)
        chosen = rng.choice(self.num_neurons, size=count, replace=False)
        self.lesion_mask[0, chosen] = 0.0
        return count

    def apply_region_lesion(
        self,
        circuit_groups: list[str],
        target_group: str,
        fraction: float,
        seed: int | None = None,
    ) -> int:
        """Silence a fraction of neurons belonging to a specific brain region ('CX'/'MB')."""
        self.reset_lesion()
        clamped = min(max(fraction, 0.0), 1.0)
        region_indices = np.array(
            [idx for idx, grp in enumerate(circuit_groups) if grp == target_group],
            dtype=np.int64,
        )
        if len(region_indices) == 0:
            return 0

        count = int(round(clamped * len(region_indices)))
        if count <= 0:
            return 0

        rng = np.random.default_rng(seed)
        chosen = rng.choice(region_indices, size=count, replace=False)
        self.lesion_mask[0, chosen] = 0.0
        return count

    def forward(
        self,
        obs: torch.Tensor,
        hidden_state: torch.Tensor,
    ) -> torch.Tensor:
        """Compute one leaky-integrator recurrent step across sparse E/I connectome."""
        mask = self.lesion_mask
        masked_hidden = hidden_state * mask

        hidden_t = masked_hidden.transpose(0, 1)
        exc_drive = torch.sparse.mm(self.w_exc_csr, hidden_t).transpose(0, 1)
        inh_drive = torch.sparse.mm(self.w_inh_csr, hidden_t).transpose(0, 1)

        synaptic_input = (
            self.g_exc * exc_drive
            - self.g_inh * inh_drive
            + self.input_projection(obs)
        )
        tau = self.leak_rate
        next_hidden = (1.0 - tau) * masked_hidden + tau * torch.tanh(synaptic_input)
        return next_hidden * mask
