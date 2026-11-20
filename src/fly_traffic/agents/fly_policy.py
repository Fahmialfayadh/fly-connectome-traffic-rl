"""Actor-critic policy built around the frozen Drosophila connectome core."""

import torch
from torch import nn

from fly_traffic.agents.base import BaseRecurrentActorCritic
from fly_traffic.agents.connectome_cell import ConnectomeRecurrentCell
from fly_traffic.connectome.artifact import ConnectomeArtifact


class FlyConnectomePolicy(BaseRecurrentActorCritic):
    """Recurrent actor-critic RL policy using the Drosophila CX/MB connectome core."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        artifact: ConnectomeArtifact,
        initial_leak_rate: float = 0.35,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.artifact = artifact

        self.cell = ConnectomeRecurrentCell(
            obs_dim=obs_dim,
            artifact=artifact,
            initial_leak_rate=initial_leak_rate,
        )
        num_neurons = artifact.metadata.num_neurons

        self.actor_head = nn.Linear(num_neurons, act_dim)
        nn.init.orthogonal_(self.actor_head.weight, gain=0.01)
        nn.init.zeros_(self.actor_head.bias)

        self.critic_head = nn.Linear(num_neurons, 1)
        nn.init.orthogonal_(self.critic_head.weight, gain=1.0)
        nn.init.zeros_(self.critic_head.bias)

    @property
    def hidden_dim(self) -> int:
        """Return number of biological neurons in the connectome reservoir."""
        return self.cell.num_neurons

    def initial_hidden(
        self,
        batch_size: int,
        device: torch.device | str = "cpu",
    ) -> torch.Tensor:
        """Create zero neural activation state [batch_size, num_neurons]."""
        return torch.zeros(
            batch_size,
            self.hidden_dim,
            dtype=torch.float32,
            device=device,
        )

    def forward_step(
        self,
        obs: torch.Tensor,
        hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Update neural state with sparse E/I dynamics and decode policy/value."""
        next_hidden = self.cell(obs, hidden)
        logits = self.actor_head(next_hidden)
        value = self.critic_head(next_hidden).squeeze(-1)
        return logits, value, next_hidden
