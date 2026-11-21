"""Parameter-matched MLP and GRU baseline policies plus checkpoint helpers."""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch import nn

from fly_traffic.agents.base import BaseRecurrentActorCritic
from fly_traffic.agents.fly_policy import FlyConnectomePolicy
from fly_traffic.config import ConnectomeConfig
from fly_traffic.connectome.artifact import ConnectomeArtifact
from fly_traffic.connectome.shuffled import load_or_create_shuffled_connectome


def _init_linear(layer: nn.Linear, gain: float) -> nn.Linear:
    """Apply orthogonal initialization with zero bias to a linear layer."""
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.zeros_(layer.bias)
    return layer


class MLPBaselinePolicy(BaseRecurrentActorCritic):
    """Parameter-matched 2-layer Tanh MLP actor-critic baseline (~69.8k params)."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_width: int = 250,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_width = hidden_width

        self.backbone = nn.Sequential(
            _init_linear(nn.Linear(obs_dim, hidden_width), gain=2.0**0.5),
            nn.Tanh(),
            _init_linear(nn.Linear(hidden_width, hidden_width), gain=2.0**0.5),
            nn.Tanh(),
        )
        self.actor_head = _init_linear(nn.Linear(hidden_width, act_dim), gain=0.01)
        self.critic_head = _init_linear(nn.Linear(hidden_width, 1), gain=1.0)
        self._last_layer1: torch.Tensor | None = None
        self._last_layer2: torch.Tensor | None = None

    @property
    def hidden_dim(self) -> int:
        """Dummy 1D hidden state placeholder so MLP shares the recurrent trainer API."""
        return 1

    def initial_hidden(
        self,
        batch_size: int,
        device: torch.device | str = "cpu",
    ) -> torch.Tensor:
        """Return dummy zero tensor [batch_size, 1]."""
        self._last_layer1 = None
        self._last_layer2 = None
        return torch.zeros(batch_size, 1, dtype=torch.float32, device=device)

    @property
    def latest_layer_activations(self) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Return detached inference activations from hidden layers 1 and 2."""
        return self._last_layer1, self._last_layer2

    def forward_step(
        self,
        obs: torch.Tensor,
        hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute policy logits and value from current observation alone."""
        layer1 = self.backbone[1](self.backbone[0](obs))
        features = self.backbone[3](self.backbone[2](layer1))
        if not torch.is_grad_enabled():
            self._last_layer1 = layer1.detach()
            self._last_layer2 = features.detach()
        logits = self.actor_head(features)
        value = self.critic_head(features).squeeze(-1)
        return logits, value, hidden

    def forward_sequence(
        self,
        obs_seq: torch.Tensor,
        initial_hidden: torch.Tensor,
        dones_seq: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Vectorized sequence evaluation across all [seq_len, batch] steps at once."""
        _ = dones_seq
        seq_len, batch_size, _ = obs_seq.shape
        flat_obs = obs_seq.reshape(seq_len * batch_size, self.obs_dim)
        features = self.backbone(flat_obs)
        logits = self.actor_head(features).reshape(seq_len, batch_size, self.act_dim)
        values = self.critic_head(features).reshape(seq_len, batch_size)
        return logits, values, initial_hidden


class GRUBaselinePolicy(BaseRecurrentActorCritic):
    """Parameter-matched recurrent GRU actor-critic baseline (~69.7k params)."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        gru_hidden_size: int = 105,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.gru_hidden_size = gru_hidden_size

        self.input_encoder = nn.Sequential(
            _init_linear(nn.Linear(obs_dim, gru_hidden_size), gain=2.0**0.5),
            nn.Tanh(),
        )
        self.gru_cell = nn.GRUCell(gru_hidden_size, gru_hidden_size)
        self.actor_head = _init_linear(nn.Linear(gru_hidden_size, act_dim), gain=0.01)
        self.critic_head = _init_linear(nn.Linear(gru_hidden_size, 1), gain=1.0)

    @property
    def hidden_dim(self) -> int:
        """Dimensionality of GRU hidden vector."""
        return self.gru_hidden_size

    def initial_hidden(
        self,
        batch_size: int,
        device: torch.device | str = "cpu",
    ) -> torch.Tensor:
        """Return zero GRU hidden state [batch_size, gru_hidden_size]."""
        return torch.zeros(
            batch_size,
            self.gru_hidden_size,
            dtype=torch.float32,
            device=device,
        )

    def forward_step(
        self,
        obs: torch.Tensor,
        hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode observation, update GRU hidden state, and decode logits and value."""
        encoded = self.input_encoder(obs)
        next_hidden = self.gru_cell(encoded, hidden)
        logits = self.actor_head(next_hidden)
        value = self.critic_head(next_hidden).squeeze(-1)
        return logits, value, next_hidden


def build_policy(
    model_type: str,
    obs_dim: int,
    act_dim: int,
    artifact: ConnectomeArtifact | None = None,
) -> BaseRecurrentActorCritic:
    """Instantiate one of the supported RL policy architectures by name."""
    normalized = model_type.lower().strip()
    if normalized in ("fly", "fly_shuffled"):
        if artifact is None:
            raise ValueError(f"ConnectomeArtifact is required to build '{normalized}' policy.")
        return FlyConnectomePolicy(obs_dim=obs_dim, act_dim=act_dim, artifact=artifact)

    if normalized == "mlp":
        return MLPBaselinePolicy(obs_dim=obs_dim, act_dim=act_dim)

    if normalized == "gru":
        return GRUBaselinePolicy(obs_dim=obs_dim, act_dim=act_dim)

    raise ValueError(
        f"Unsupported model_type '{model_type}'. Expected fly, fly_shuffled, mlp, or gru."
    )


def save_policy_checkpoint(
    path: Path,
    model_type: str,
    policy: BaseRecurrentActorCritic,
    obs_dim: int,
    act_dim: int,
    history: dict[str, list[float]] | None = None,
    connectome_path: Path | None = None,
    trained_timesteps: int | None = None,
) -> None:
    """Save trained policy state dict and metadata for reproducible reload."""
    path.parent.mkdir(parents=True, exist_ok=True)
    history_dict = history or {}
    inferred_timesteps = (
        int(history_dict["timesteps"][-1])
        if history_dict.get("timesteps")
        else 0
    )
    resolved_timesteps = (
        int(trained_timesteps) if trained_timesteps is not None else inferred_timesteps
    )
    leak_rate_tau = (
        float(policy.cell.leak_rate.item())
        if isinstance(policy, FlyConnectomePolicy)
        else None
    )

    payload: dict[str, Any] = {
        "model_type": model_type,
        "obs_dim": int(obs_dim),
        "act_dim": int(act_dim),
        "trainable_params": int(policy.count_trainable_parameters()),
        "trained_timesteps": resolved_timesteps,
        "leak_rate_tau": leak_rate_tau,
        "saved_at_iso": datetime.now(timezone.utc).isoformat(),
        "state_dict": policy.state_dict(),
        "history": history_dict,
        "connectome_path": str(connectome_path) if connectome_path else None,
    }
    torch.save(payload, path)


def load_policy_checkpoint(
    path: Path,
    artifact: ConnectomeArtifact | None = None,
    device: torch.device | str = "cpu",
    shuffled_artifact: ConnectomeArtifact | None = None,
) -> tuple[BaseRecurrentActorCritic, dict[str, Any]]:
    """Reload a saved policy checkpoint onto CPU/target device."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    model_type = str(payload["model_type"])
    obs_dim = int(payload["obs_dim"])
    act_dim = int(payload["act_dim"])

    resolved_artifact = artifact
    if model_type == "fly_shuffled":
        resolved_artifact = shuffled_artifact
        if resolved_artifact is None:
            conn_path_raw = payload.get("connectome_path")
            if conn_path_raw and Path(conn_path_raw).exists():
                resolved_artifact = ConnectomeArtifact.load(Path(conn_path_raw))
            else:
                resolved_artifact = load_or_create_shuffled_connectome()
    elif model_type == "fly" and resolved_artifact is None:
        conn_path_raw = payload.get("connectome_path")
        candidate = Path(conn_path_raw) if conn_path_raw else ConnectomeConfig().output_path
        resolved_artifact = ConnectomeArtifact.load(candidate)

    policy = build_policy(
        model_type=model_type,
        obs_dim=obs_dim,
        act_dim=act_dim,
        artifact=resolved_artifact,
    )
    policy.load_state_dict(payload["state_dict"], strict=True)
    policy.to(device)
    policy.eval()
    return policy, payload
