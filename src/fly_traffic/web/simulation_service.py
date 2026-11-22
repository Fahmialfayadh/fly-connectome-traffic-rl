"""Thread-safe live simulation engine and neural telemetry encoder for the Web UI."""

import base64
from datetime import datetime, timezone
from pathlib import Path
import threading
from typing import Any

import numpy as np
import torch

from fly_traffic.agents.base import BaseTrafficAgent, NeuralAgentAdapter
from fly_traffic.agents.baselines import (
    GRUBaselinePolicy,
    MLPBaselinePolicy,
    load_policy_checkpoint,
)
from fly_traffic.agents.fixed_timer import FixedTimerAgent
from fly_traffic.agents.fly_policy import FlyConnectomePolicy
from fly_traffic.config import ConnectomeConfig, PPOConfig, TrafficEnvConfig
from fly_traffic.connectome.artifact import ConnectomeArtifact
from fly_traffic.connectome.shuffled import load_or_create_shuffled_connectome
from fly_traffic.environment.traffic_env import TrafficIntersectionEnv


class SimulationService:
    """Manages live inference state, checkpoint provenance, and connectome perturbations."""

    AGENT_LABELS: dict[str, str] = {
        "fly": "Fly Connectome PPO",
        "fly_shuffled": "Shuffled Connectome PPO",
        "gru": "PPO-GRU",
        "mlp": "PPO-MLP",
        "fixed_timer": "Fixed Timer",
    }
    DEFAULT_MODEL_KEYS: tuple[str, ...] = ("fly", "fly_shuffled", "gru", "mlp")
    DEFAULT_CHECKPOINT_FILES: frozenset[str] = frozenset(
        f"{key}_ppo.pt" for key in DEFAULT_MODEL_KEYS
    )
    DEFAULT_CHECKPOINT_TO_KEY: dict[str, str] = {
        f"{key}_ppo.pt": key for key in DEFAULT_MODEL_KEYS
    }

    def __init__(
        self,
        env_config: TrafficEnvConfig | None = None,
        connectome_path: Path | None = None,
        checkpoints_dir: Path | None = None,
    ) -> None:
        self.env_config = env_config or TrafficEnvConfig()
        self.connectome_path = connectome_path or ConnectomeConfig.output_path
        self.checkpoints_dir = checkpoints_dir or PPOConfig.checkpoints_dir

        self._lock = threading.Lock()
        self._running: bool = True
        self._steps_per_second: float = 2.0
        self._scenario_name: str = "rush_hour"
        self._pending_episode_reset: bool = False
        self._episode_just_ended: bool = False
        self._last_chosen_action: int = 0
        self._spawn_rng = np.random.default_rng()
        self._episode_history: list[dict[str, Any]] = []
        self._prev_gru_hidden: np.ndarray | None = None

        self._artifact: ConnectomeArtifact | None = (
            ConnectomeArtifact.load(self.connectome_path)
            if self.connectome_path.exists()
            else None
        )
        self._shuffled_artifact: ConnectomeArtifact | None = (
            load_or_create_shuffled_connectome(self.connectome_path)
            if self.connectome_path.exists()
            else None
        )
        self._env = TrafficIntersectionEnv(
            config=self.env_config,
            profile_name=self._scenario_name,
            continuous_mode=True,
        )
        self._env.set_lane_capacity_limit(self.env_config.hard_max_lane_capacity)
        self._current_obs, _ = self._env.reset(seed=42)

        self._agents: dict[str, BaseTrafficAgent] = {}
        self._checkpoint_inventory: dict[str, dict[str, Any]] = {}
        self._checkpoint_mtimes: dict[str, int] = {}
        self._active_agent_key: str = ""
        self._load_all_agents()

        self._active_agent_key = self._pick_default_agent_key()
        self._agents[self._active_agent_key].reset()

    @property
    def is_running(self) -> bool:
        """Whether automatic continuous stepping is active."""
        with self._lock:
            return self._running

    @property
    def steps_per_second(self) -> float:
        """Current target simulation decision frequency (ticks/second)."""
        with self._lock:
            return self._steps_per_second

    def step_once(self) -> dict[str, Any]:
        """Advance the traffic simulation by one discrete interval and return frame data."""
        with self._lock:
            active_agent = self._agents[self._active_agent_key]

            if self._pending_episode_reset:
                self._current_obs, _ = self._env.reset(
                    options={"profile_name": self._scenario_name}
                )
                active_agent.reset()
                self._prev_gru_hidden = None
                self._pending_episode_reset = False

            if isinstance(active_agent, NeuralAgentAdapter) and isinstance(
                active_agent.policy,
                GRUBaselinePolicy,
            ):
                self._prev_gru_hidden = active_agent.latest_hidden_numpy.copy()

            chosen_action = active_agent.select_action(
                self._current_obs,
                deterministic=True,
            )
            self._last_chosen_action = int(chosen_action)

            next_obs, _reward, terminated, truncated, _info = self._env.step(chosen_action)
            self._current_obs = next_obs

            episode_finished = bool(terminated or truncated)
            self._episode_just_ended = episode_finished
            if episode_finished:
                self._pending_episode_reset = True
                ep_summary = {
                    "episode_reward": float(self._env._episode_reward),
                    "total_throughput": int(self._env._total_throughput),
                    "total_switches": int(self._env._total_switches),
                    "total_collisions": int(self._env._total_collisions),
                    "steps": int(self._env._current_step),
                    "scenario": self._scenario_name,
                    "agent_key": self._active_agent_key,
                }
                self._episode_history.append(ep_summary)
                if len(self._episode_history) > 50:
                    self._episode_history.pop(0)

            return self._build_frame_payload_unlocked()

    def reset_simulation(self, seed: int | None = 42) -> dict[str, Any]:
        """Reset intersection queues, vehicle state, and agent recurrent memory."""
        with self._lock:
            self._current_obs, _ = self._env.reset(
                seed=seed,
                options={"profile_name": self._scenario_name},
            )
            self._agents[self._active_agent_key].reset()
            self._prev_gru_hidden = None
            self._pending_episode_reset = False
            self._episode_just_ended = False
            self._last_chosen_action = 0
            self._episode_history = []
            return self._build_frame_payload_unlocked()

    def set_running(self, running: bool) -> dict[str, Any]:
        """Play or pause continuous simulation stepping."""
        with self._lock:
            self._running = bool(running)
            return self._build_frame_payload_unlocked()

    def set_speed(self, steps_per_second: float) -> dict[str, Any]:
        """Update simulation stepping cadence within [0.5, 10.0] steps/sec."""
        with self._lock:
            self._steps_per_second = min(max(float(steps_per_second), 0.5), 10.0)
            return self._build_frame_payload_unlocked()

    def set_scenario(self, scenario_name: str) -> dict[str, Any]:
        """Switch traffic arrival profile ('normal', 'asymmetric', or 'rush_hour')."""
        with self._lock:
            if scenario_name not in ("normal", "asymmetric", "rush_hour"):
                raise ValueError(f"Unsupported scenario '{scenario_name}'.")
            self._scenario_name = scenario_name
            self._env.set_profile(scenario_name)
            return self._build_frame_payload_unlocked()

    def set_safety_buffer(self, all_red_steps: int) -> dict[str, Any]:
        """Update intergreen ALL_RED clearance duration (steps) on the live environment."""
        with self._lock:
            self._env.set_safety_buffer(all_red_steps)
            return self._build_frame_payload_unlocked()

    def set_continuous_mode(self, enabled: bool) -> dict[str, Any]:
        """Toggle endless simulation mode vs fixed max_steps episode truncation."""
        with self._lock:
            is_enabled = bool(enabled)
            self._env.set_continuous_mode(is_enabled)
            if is_enabled:
                self._pending_episode_reset = False
                self._episode_just_ended = False
            return self._build_frame_payload_unlocked()

    def spawn_vehicles(self, lane_id: int, count: int) -> dict[str, Any]:
        """Manually inject clamped vehicles into a single validated lane (0..7)."""
        if isinstance(lane_id, bool):
            raise ValueError("lane_id must be an integer in 0..7.")
        normalized_lane = int(lane_id)
        if not (0 <= normalized_lane < TrafficIntersectionEnv.NUM_LANES):
            raise ValueError(
                f"lane_id {normalized_lane} must be in 0..{TrafficIntersectionEnv.NUM_LANES - 1}."
            )
        if isinstance(count, bool):
            raise ValueError("count must be an integer.")
        clamped_count = min(max(int(count), 1), 40)

        with self._lock:
            self._env.inject_vehicles({normalized_lane: clamped_count})
            self._current_obs = self._env._build_observation()
            return self._build_frame_payload_unlocked()

    def spawn_random_vehicles(self, count: int = 5) -> dict[str, Any]:
        """Randomly distribute clamped vehicles across all 8 lanes."""
        if isinstance(count, bool):
            raise ValueError("count must be an integer.")
        clamped_count = min(max(int(count), 1), 80)

        with self._lock:
            sampled_lanes = self._spawn_rng.integers(
                0,
                TrafficIntersectionEnv.NUM_LANES,
                size=clamped_count,
            )
            lane_counts: dict[int, int] = {}
            for lane_index in sampled_lanes:
                lane_key = int(lane_index)
                lane_counts[lane_key] = lane_counts.get(lane_key, 0) + 1
            self._env.inject_vehicles(lane_counts)
            self._current_obs = self._env._build_observation()
            return self._build_frame_payload_unlocked()

    def spawn_surge(self, count_per_lane: int = 3) -> dict[str, Any]:
        """Inject a balanced multi-lane traffic surge across all 8 incoming lanes."""
        if isinstance(count_per_lane, bool):
            raise ValueError("count_per_lane must be an integer.")
        clamped_per_lane = min(max(int(count_per_lane), 1), 20)
        lane_counts = {
            lane_index: clamped_per_lane
            for lane_index in range(TrafficIntersectionEnv.NUM_LANES)
        }

        with self._lock:
            self._env.inject_vehicles(lane_counts)
            self._current_obs = self._env._build_observation()
            return self._build_frame_payload_unlocked()

    def set_agent(self, agent_key: str) -> dict[str, Any]:
        """Switch the active inference controller and reset its recurrent hidden state."""
        with self._lock:
            if agent_key not in self._agents:
                raise ValueError(
                    f"Agent '{agent_key}' is not loaded. Available: {list(self._agents)}"
                )
            self._active_agent_key = agent_key
            self._agents[agent_key].reset()
            self._prev_gru_hidden = None
            return self._build_frame_payload_unlocked()

    def set_pharmacology(self, g_exc: float, g_inh: float) -> dict[str, Any]:
        """Apply live excitatory (ACh) and inhibitory (GABA/Glu) gain scaling."""
        with self._lock:
            fly_policy = self._get_fly_policy_unlocked()
            if fly_policy is not None:
                fly_policy.cell.set_pharmacology(g_exc=g_exc, g_inh=g_inh)
            return self._build_frame_payload_unlocked()

    def apply_lesion(self, target: str, fraction: float = 0.25) -> dict[str, Any]:
        """Silence a fraction of CX, MB, or random neurons on the active fly core."""
        with self._lock:
            fly_policy = self._get_fly_policy_unlocked()
            if fly_policy is not None:
                normalized = target.strip().upper()
                if normalized in ("CX", "MB"):
                    fly_policy.cell.apply_region_lesion(
                        circuit_groups=fly_policy.artifact.circuit_groups,
                        target_group=normalized,
                        fraction=fraction,
                        seed=42,
                    )
                elif normalized == "RANDOM":
                    fly_policy.cell.apply_random_lesion(fraction=fraction, seed=42)
                else:
                    raise ValueError(f"Unknown lesion target '{target}'.")
            return self._build_frame_payload_unlocked()

    def reset_brain(self) -> dict[str, Any]:
        """Restore pharmacology gains to (1.0, 1.0) and clear all neural lesions."""
        with self._lock:
            fly_policy = self._get_fly_policy_unlocked()
            if fly_policy is not None:
                fly_policy.cell.set_pharmacology(1.0, 1.0)
                fly_policy.cell.reset_lesion()
            return self._build_frame_payload_unlocked()

    def reload_checkpoint(self, model_type: str) -> dict[str, Any]:
        """Reload a freshly trained checkpoint from disk after Training Studio finishes."""
        with self._lock:
            normalized = model_type.lower().strip()
            self._load_single_neural_agent_unlocked(normalized)
            self._refresh_inventory_order_unlocked()
            if normalized in self._agents:
                self._active_agent_key = normalized
                self._agents[normalized].reset()
            return self._build_frame_payload_unlocked()

    def register_checkpoint(
        self,
        checkpoint_path: Path,
        activate: bool = True,
    ) -> tuple[str, dict[str, Any]]:
        """Register or reload a saved policy checkpoint and optionally activate it."""
        with self._lock:
            default_key = self.DEFAULT_CHECKPOINT_TO_KEY.get(checkpoint_path.name)
            if default_key is not None:
                self._load_single_neural_agent_unlocked(
                    default_key,
                    checkpoint_path=checkpoint_path,
                )
                resolved_key = default_key
            else:
                resolved_key = self._load_custom_checkpoint_unlocked(checkpoint_path)

            self._refresh_inventory_order_unlocked()

            if activate and resolved_key in self._agents:
                self._active_agent_key = resolved_key
                self._agents[resolved_key].reset()

            return resolved_key, self._build_frame_payload_unlocked()

    def get_frame_payload(self) -> dict[str, Any]:
        """Return a thread-safe snapshot of the current simulation and connectome state."""
        with self._lock:
            return self._build_frame_payload_unlocked()

    def get_overview_payload(self) -> dict[str, Any]:
        """Return connectome summary and checkpoint inventory for UI provenance."""
        with self._lock:
            self._sync_checkpoints_with_disk_unlocked()
            connectome_summary = (
                self._artifact.metadata.to_dict() if self._artifact is not None else None
            )
            return {
                "connectome_path": str(self.connectome_path),
                "connectome_metadata": connectome_summary,
                "checkpoints": dict(self._checkpoint_inventory),
                "active_agent_key": self._active_agent_key,
            }

    def _sync_checkpoints_with_disk_unlocked(self) -> None:
        """Reload any default or custom .pt checkpoints created/updated on disk."""
        changed = False
        for model_key in self.DEFAULT_MODEL_KEYS:
            default_path = self.checkpoints_dir / f"{model_key}_ppo.pt"
            if default_path.is_file():
                mtime_ns = int(default_path.stat().st_mtime_ns)
                if self._checkpoint_mtimes.get(model_key) != mtime_ns:
                    try:
                        self._load_single_neural_agent_unlocked(model_key)
                        changed = True
                    except Exception:
                        continue
            elif self._checkpoint_inventory.get(model_key, {}).get("exists"):
                self._checkpoint_mtimes.pop(model_key, None)
                self._load_single_neural_agent_unlocked(model_key)
                changed = True

        seen_custom_keys: set[str] = set()
        if self.checkpoints_dir.exists():
            for checkpoint_path in sorted(self.checkpoints_dir.glob("*.pt")):
                if (
                    not checkpoint_path.is_file()
                    or checkpoint_path.name in self.DEFAULT_CHECKPOINT_FILES
                ):
                    continue
                agent_key = f"ckpt:{checkpoint_path.stem}"
                seen_custom_keys.add(agent_key)
                mtime_ns = int(checkpoint_path.stat().st_mtime_ns)
                if self._checkpoint_mtimes.get(agent_key) != mtime_ns:
                    try:
                        self._load_custom_checkpoint_unlocked(checkpoint_path)
                        changed = True
                    except Exception:
                        continue

        stale_custom_keys = [
            key
            for key in list(self._checkpoint_inventory.keys())
            if key.startswith("ckpt:")
            and key not in seen_custom_keys
            and key != self._active_agent_key
        ]
        for stale_key in stale_custom_keys:
            self._checkpoint_inventory.pop(stale_key, None)
            self._checkpoint_mtimes.pop(stale_key, None)
            self._agents.pop(stale_key, None)
            changed = True

        if changed:
            self._refresh_inventory_order_unlocked()

    def _load_all_agents(self) -> None:
        """Load default checkpoints, custom .pt policies, and the rule-based Fixed Timer."""
        for model_key in self.DEFAULT_MODEL_KEYS:
            self._load_single_neural_agent_unlocked(model_key)

        if self.checkpoints_dir.exists():
            for checkpoint_path in sorted(self.checkpoints_dir.glob("*.pt")):
                if (
                    not checkpoint_path.is_file()
                    or checkpoint_path.name in self.DEFAULT_CHECKPOINT_FILES
                ):
                    continue
                try:
                    self._load_custom_checkpoint_unlocked(checkpoint_path)
                except Exception:
                    continue

        self._agents["fixed_timer"] = FixedTimerAgent()
        self._checkpoint_inventory["fixed_timer"] = {
            "key": "fixed_timer",
            "model_type": "fixed_timer",
            "label": self.AGENT_LABELS["fixed_timer"],
            "checkpoint_file": "Rule-based cyclic timer (no learned weights)",
            "relative_path": "rule-based",
            "is_custom": False,
            "exists": True,
            "trainable_params": 0,
            "trained_timesteps": 0,
            "saved_at_iso": None,
        }
        self._refresh_inventory_order_unlocked()

    def _load_single_neural_agent_unlocked(
        self,
        model_key: str,
        checkpoint_path: Path | None = None,
    ) -> None:
        resolved_path = checkpoint_path or (self.checkpoints_dir / f"{model_key}_ppo.pt")
        display_label = self.AGENT_LABELS[model_key]

        if resolved_path.exists():
            policy, payload = load_policy_checkpoint(
                path=resolved_path,
                artifact=self._artifact if model_key == "fly" else None,
                shuffled_artifact=self._shuffled_artifact if model_key == "fly_shuffled" else None,
            )
            trained_steps, saved_at = self._extract_checkpoint_metadata(
                resolved_path,
                payload,
            )

            self._agents[model_key] = NeuralAgentAdapter(
                policy=policy,
                agent_name=display_label,
            )
            self._checkpoint_inventory[model_key] = {
                "key": model_key,
                "model_type": model_key,
                "label": display_label,
                "checkpoint_file": str(resolved_path.name),
                "relative_path": f"artifacts/checkpoints/{resolved_path.name}",
                "is_custom": False,
                "exists": True,
                "trainable_params": int(policy.count_trainable_parameters()),
                "trained_timesteps": trained_steps,
                "saved_at_iso": saved_at,
            }
            self._checkpoint_mtimes[model_key] = int(resolved_path.stat().st_mtime_ns)
            return

        if model_key in ("fly", "fly_shuffled"):
            target_artifact = (
                self._shuffled_artifact if model_key == "fly_shuffled" else self._artifact
            )
            if target_artifact is not None:
                untrained_policy = FlyConnectomePolicy(
                    obs_dim=22,
                    act_dim=4,
                    artifact=target_artifact,
                )
                self._agents[model_key] = NeuralAgentAdapter(
                    policy=untrained_policy,
                    agent_name=f"{display_label} (untrained)",
                )
                self._checkpoint_inventory[model_key] = {
                    "key": model_key,
                    "model_type": model_key,
                    "label": f"{display_label} (untrained)",
                    "checkpoint_file": "No checkpoint saved yet",
                    "relative_path": "Not saved yet",
                    "is_custom": False,
                    "exists": False,
                    "trainable_params": int(untrained_policy.count_trainable_parameters()),
                    "trained_timesteps": 0,
                    "saved_at_iso": None,
                }
                return

        self._checkpoint_inventory[model_key] = {
            "key": model_key,
            "model_type": model_key,
            "label": display_label,
            "checkpoint_file": "Not trained yet",
            "relative_path": "Not saved yet",
            "is_custom": False,
            "exists": False,
            "trainable_params": 0,
            "trained_timesteps": 0,
            "saved_at_iso": None,
        }

    def _load_custom_checkpoint_unlocked(self, checkpoint_path: Path) -> str:
        """Load and register a custom .pt policy file under key 'ckpt:<stem>'."""
        policy, payload = load_policy_checkpoint(
            path=checkpoint_path,
            artifact=self._artifact,
            shuffled_artifact=self._shuffled_artifact,
        )
        if not isinstance(payload, dict):
            raise ValueError(f"Checkpoint '{checkpoint_path.name}' payload is not a dict.")

        loaded_model_type = str(payload.get("model_type", "")).lower().strip()
        if loaded_model_type not in self.DEFAULT_MODEL_KEYS:
            raise ValueError(
                f"Unsupported checkpoint model_type '{loaded_model_type}' in '{checkpoint_path.name}'."
            )

        expected_obs_dim = int(self._env.observation_space.shape[0])
        expected_act_dim = int(self._env.action_space.n)
        if (
            int(payload.get("obs_dim", -1)) != expected_obs_dim
            or int(payload.get("act_dim", -1)) != expected_act_dim
        ):
            raise ValueError(
                f"Incompatible policy dimensions in '{checkpoint_path.name}'."
            )

        trained_steps, saved_at = self._extract_checkpoint_metadata(
            checkpoint_path,
            payload,
        )
        agent_key = f"ckpt:{checkpoint_path.stem}"
        display_label = f"{self.AGENT_LABELS[loaded_model_type]} ({checkpoint_path.stem})"

        self._agents[agent_key] = NeuralAgentAdapter(
            policy=policy,
            agent_name=display_label,
        )
        self._checkpoint_inventory[agent_key] = {
            "key": agent_key,
            "model_type": loaded_model_type,
            "label": display_label,
            "checkpoint_file": checkpoint_path.name,
            "relative_path": f"artifacts/checkpoints/{checkpoint_path.name}",
            "is_custom": True,
            "exists": True,
            "trainable_params": int(policy.count_trainable_parameters()),
            "trained_timesteps": trained_steps,
            "saved_at_iso": saved_at,
        }
        self._checkpoint_mtimes[agent_key] = int(checkpoint_path.stat().st_mtime_ns)
        return agent_key

    @staticmethod
    def _extract_checkpoint_metadata(
        checkpoint_path: Path,
        payload: dict[str, Any],
    ) -> tuple[int, str]:
        """Extract trained timesteps and UTC ISO timestamp from checkpoint payload or stat."""
        history = payload.get("history") or {}
        timesteps_list = history.get("timesteps") or []
        trained_steps = int(
            payload.get("trained_timesteps")
            or (timesteps_list[-1] if timesteps_list else 0)
        )
        saved_at = payload.get("saved_at_iso")
        if not saved_at:
            mtime = datetime.fromtimestamp(
                checkpoint_path.stat().st_mtime,
                tz=timezone.utc,
            )
            saved_at = mtime.isoformat()
        return trained_steps, str(saved_at)

    def _refresh_inventory_order_unlocked(self) -> None:
        """Keep inventory ordered: defaults (fly, gru, mlp) -> sorted ckpt:* -> fixed_timer."""
        ordered_keys: list[str] = [
            key for key in self.DEFAULT_MODEL_KEYS if key in self._checkpoint_inventory
        ]
        ordered_keys.extend(
            sorted(key for key in self._checkpoint_inventory if key.startswith("ckpt:"))
        )
        if "fixed_timer" in self._checkpoint_inventory:
            ordered_keys.append("fixed_timer")

        self._checkpoint_inventory = {
            key: self._checkpoint_inventory[key] for key in ordered_keys
        }
        ordered_agents: dict[str, BaseTrafficAgent] = {}
        for key in ordered_keys:
            if key in self._agents:
                ordered_agents[key] = self._agents[key]
        for key, agent in self._agents.items():
            if key not in ordered_agents:
                ordered_agents[key] = agent
        self._agents = ordered_agents

    def _pick_default_agent_key(self) -> str:
        for candidate in ("fly", "gru", "mlp", "fixed_timer"):
            if candidate in self._agents:
                return candidate
        return "fixed_timer"

    def _get_fly_policy_unlocked(self) -> FlyConnectomePolicy | None:
        active_agent = self._agents.get(self._active_agent_key)
        if isinstance(active_agent, NeuralAgentAdapter) and isinstance(
            active_agent.policy,
            FlyConnectomePolicy,
        ):
            return active_agent.policy

        fly_agent = self._agents.get("fly")
        if isinstance(fly_agent, NeuralAgentAdapter) and isinstance(
            fly_agent.policy,
            FlyConnectomePolicy,
        ):
            return fly_agent.policy
        return None

    def _build_frame_payload_unlocked(self) -> dict[str, Any]:
        active_agent = self._agents[self._active_agent_key]
        snapshot = self._env.get_render_snapshot()
        checkpoint_info = self._checkpoint_inventory.get(self._active_agent_key, {})

        if isinstance(active_agent, NeuralAgentAdapter):
            action_probs = [float(p) for p in active_agent.latest_action_probabilities]
            value_estimate: float | None = float(active_agent.latest_value_estimate)
        else:
            action_probs = [
                1.0 if idx == self._last_chosen_action else 0.0
                for idx in range(4)
            ]
            value_estimate = None

        fly_policy = self._get_fly_policy_unlocked()
        is_fly_active = (
            isinstance(active_agent, NeuralAgentAdapter)
            and isinstance(active_agent.policy, FlyConnectomePolicy)
        )

        if is_fly_active and fly_policy is not None:
            hidden_np = active_agent.latest_hidden_numpy
            lesion_mask_np = (
                fly_policy.cell.lesion_mask.squeeze(0).detach().cpu().numpy()
            )
            cx_count = fly_policy.artifact.metadata.cx_count
            cx_mean_abs = float(np.mean(np.abs(hidden_np[:cx_count])))
            mb_mean_abs = float(np.mean(np.abs(hidden_np[cx_count:])))

            # Quantize [-0.4, +0.4] neural activations into 0..255 for compact WS transport.
            scaled_u8 = np.clip(
                np.round(((hidden_np / 0.4) * 0.5 + 0.5) * 255.0),
                0.0,
                255.0,
            ).astype(np.uint8)
            lesion_u8 = (lesion_mask_np > 0.5).astype(np.uint8)

            connectome_telemetry = {
                "is_fly_active": True,
                "g_exc": float(fly_policy.cell.g_exc),
                "g_inh": float(fly_policy.cell.g_inh),
                "leak_rate_tau": float(fly_policy.cell.leak_rate.item()),
                "active_neurons": int(fly_policy.cell.active_neuron_count()),
                "total_neurons": int(fly_policy.cell.num_neurons),
                "cx_mean_abs_activity": cx_mean_abs,
                "mb_mean_abs_activity": mb_mean_abs,
                "activations_b64": base64.b64encode(scaled_u8.tobytes()).decode("ascii"),
                "lesion_mask_b64": base64.b64encode(lesion_u8.tobytes()).decode("ascii"),
            }
        else:
            connectome_telemetry = {
                "is_fly_active": False,
                "g_exc": float(fly_policy.cell.g_exc) if fly_policy else 1.0,
                "g_inh": float(fly_policy.cell.g_inh) if fly_policy else 1.0,
                "leak_rate_tau": (
                    float(fly_policy.cell.leak_rate.item()) if fly_policy else 0.0
                ),
                "active_neurons": (
                    int(fly_policy.cell.active_neuron_count()) if fly_policy else 0
                ),
                "total_neurons": int(fly_policy.cell.num_neurons) if fly_policy else 0,
                "cx_mean_abs_activity": 0.0,
                "mb_mean_abs_activity": 0.0,
                "activations_b64": None,
                "lesion_mask_b64": None,
            }

        model_telemetry = self._build_model_telemetry_unlocked(
            active_agent=active_agent,
            checkpoint_info=checkpoint_info,
            snapshot=snapshot,
        )

        return {
            "type": "simulation_frame",
            "mode": "INFERENCE",
            "running": self._running,
            "steps_per_second": self._steps_per_second,
            "agent_key": self._active_agent_key,
            "agent_name": active_agent.name,
            "scenario": self._scenario_name,
            "episode_just_ended": self._episode_just_ended,
            "checkpoint_provenance": checkpoint_info,
            "snapshot": snapshot,
            "policy_telemetry": {
                "action_probabilities": action_probs,
                "chosen_action": self._last_chosen_action,
                "value_estimate": value_estimate,
            },
            "connectome_telemetry": connectome_telemetry,
            "model_telemetry": model_telemetry,
            "episode_history": list(self._episode_history),
        }

    @staticmethod
    def _quantize_unit_interval_b64(values: np.ndarray) -> str:
        """Encode float activations in [-1, 1] into compact uint8 base64."""
        scaled = np.clip(np.round((values * 0.5 + 0.5) * 255.0), 0.0, 255.0).astype(
            np.uint8
        )
        return base64.b64encode(scaled.tobytes()).decode("ascii")

    def _build_model_telemetry_unlocked(
        self,
        active_agent: BaseTrafficAgent,
        checkpoint_info: dict[str, Any],
        snapshot: dict[str, Any],
    ) -> dict[str, Any]:
        """Build factual live telemetry specific to the currently active controller."""
        model_type = str(checkpoint_info.get("model_type") or self._active_agent_key)
        trainable_params = int(checkpoint_info.get("trainable_params") or 0)
        trained_timesteps = int(checkpoint_info.get("trained_timesteps") or 0)
        checkpoint_file = str(checkpoint_info.get("checkpoint_file") or "-")

        if isinstance(active_agent, NeuralAgentAdapter) and isinstance(
            active_agent.policy,
            FlyConnectomePolicy,
        ):
            meta = active_agent.policy.artifact.metadata
            is_shuffled = model_type == "fly_shuffled"
            return {
                "kind": "fly_shuffled" if is_shuffled else "fly",
                "is_shuffled": is_shuffled,
                "architecture": (
                    "FlyConnectomePolicy (Shuffled Null Model)"
                    if is_shuffled
                    else "FlyConnectomePolicy (FAFB v783)"
                ),
                "recurrent": True,
                "trainable_params": trainable_params,
                "trained_timesteps": trained_timesteps,
                "checkpoint_file": checkpoint_file,
                "num_neurons": int(meta.num_neurons),
                "cx_count": int(meta.cx_count),
                "mb_count": int(meta.mb_count),
                "exc_edges": int(meta.exc_edge_count),
                "inh_edges": int(meta.inh_edge_count),
                "cross_region_edges": int(meta.cross_region_edges),
                "sparsity": float(meta.sparsity),
                "spectral_radius": float(meta.scaled_spectral_radius),
            }

        if isinstance(active_agent, NeuralAgentAdapter) and isinstance(
            active_agent.policy,
            GRUBaselinePolicy,
        ):
            hidden_np = active_agent.latest_hidden_numpy
            abs_h = np.abs(hidden_np)
            if (
                self._prev_gru_hidden is not None
                and self._prev_gru_hidden.shape == hidden_np.shape
            ):
                delta_l2 = float(np.linalg.norm(hidden_np - self._prev_gru_hidden))
            else:
                delta_l2 = 0.0
            return {
                "kind": "gru",
                "architecture": f"GRUBaselinePolicy ({active_agent.policy.gru_hidden_size} GRU units)",
                "recurrent": True,
                "trainable_params": trainable_params,
                "trained_timesteps": trained_timesteps,
                "checkpoint_file": checkpoint_file,
                "hidden_dim": int(hidden_np.shape[0]),
                "activations_b64": self._quantize_unit_interval_b64(hidden_np),
                "mean_abs": float(np.mean(abs_h)),
                "std": float(np.std(hidden_np)),
                "max_abs": float(np.max(abs_h)) if hidden_np.size > 0 else 0.0,
                "l2_norm": float(np.linalg.norm(hidden_np)),
                "delta_l2": delta_l2,
                "positive_ratio": float(np.mean(hidden_np > 0.0)),
                "saturated_ratio": float(np.mean(abs_h > 0.8)),
            }

        if isinstance(active_agent, NeuralAgentAdapter) and isinstance(
            active_agent.policy,
            MLPBaselinePolicy,
        ):
            l1_t, l2_t = active_agent.policy.latest_layer_activations
            width = int(active_agent.policy.hidden_width)
            l1_np = (
                l1_t.squeeze(0).detach().cpu().numpy().astype(np.float32)
                if l1_t is not None
                else np.zeros(width, dtype=np.float32)
            )
            l2_np = (
                l2_t.squeeze(0).detach().cpu().numpy().astype(np.float32)
                if l2_t is not None
                else np.zeros(width, dtype=np.float32)
            )
            l1_abs = np.abs(l1_np)
            l2_abs = np.abs(l2_np)
            return {
                "kind": "mlp",
                "architecture": f"MLPBaselinePolicy (2x{width} Tanh)",
                "recurrent": False,
                "trainable_params": trainable_params,
                "trained_timesteps": trained_timesteps,
                "checkpoint_file": checkpoint_file,
                "hidden_width": width,
                "layer1_b64": self._quantize_unit_interval_b64(l1_np),
                "layer2_b64": self._quantize_unit_interval_b64(l2_np),
                "layer1_mean_abs": float(np.mean(l1_abs)),
                "layer1_std": float(np.std(l1_np)),
                "layer1_saturated_ratio": float(np.mean(l1_abs > 0.8)),
                "layer2_mean_abs": float(np.mean(l2_abs)),
                "layer2_std": float(np.std(l2_np)),
                "layer2_saturated_ratio": float(np.mean(l2_abs > 0.8)),
                "layer2_l2_norm": float(np.linalg.norm(l2_np)),
            }

        phase_durations = (
            list(active_agent.phase_durations)
            if isinstance(active_agent, FixedTimerAgent)
            else [8, 4, 8, 4]
        )
        active_phase = int(snapshot.get("active_phase", 0))
        phase_elapsed = int(snapshot.get("phase_elapsed", 0))
        signal_stage = str(snapshot.get("signal_stage", "GREEN"))
        target_steps = int(phase_durations[active_phase % len(phase_durations)])
        remaining_steps = max(0, target_steps - phase_elapsed)
        return {
            "kind": "fixed_timer",
            "architecture": "FixedTimerAgent (8-4-8-4 cyclic schedule)",
            "recurrent": False,
            "trainable_params": 0,
            "trained_timesteps": 0,
            "checkpoint_file": "Rule-based",
            "phase_durations": phase_durations,
            "active_phase": active_phase,
            "next_phase": (active_phase + 1) % 4,
            "phase_elapsed": phase_elapsed,
            "phase_target_steps": target_steps,
            "remaining_steps": remaining_steps,
            "is_transitioning": signal_stage != "GREEN",
            "signal_stage": signal_stage,
            "dt_seconds": float(self.env_config.dt_seconds),
            "cycle_total_steps": int(sum(phase_durations)),
        }
