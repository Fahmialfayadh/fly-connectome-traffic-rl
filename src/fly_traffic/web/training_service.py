"""Background CPU PPO training job manager with real-time WebSocket streaming."""

import asyncio
from collections.abc import Callable
from datetime import datetime
import math
from pathlib import Path
import re
import threading
from typing import Any

import numpy as np
import torch

from fly_traffic.agents.base import BaseRecurrentActorCritic
from fly_traffic.config import ConnectomeConfig, PPOConfig, TrafficEnvConfig
from fly_traffic.connectome.artifact import ConnectomeArtifact
from fly_traffic.connectome.shuffled import (
    default_shuffled_connectome_path,
    load_or_create_shuffled_connectome,
)
from fly_traffic.environment.traffic_env import TrafficIntersectionEnv
from fly_traffic.training.ppo_trainer import PPOTrainer
from fly_traffic.web.simulation_service import SimulationService


class TrainingJobManager:
    """Executes PPO training on a worker thread and streams per-update metrics."""

    VALID_SAVE_MODES: tuple[str, ...] = (
        "new_file",
        "overwrite_default",
        "overwrite_resumed",
        "none",
    )
    VALID_ROLLOUT_STEPS: tuple[int, ...] = (32, 64, 128, 256)
    VALID_SEQUENCE_CHUNK_LENS: tuple[int, ...] = (8, 16, 32)
    VALID_NUM_MINIBATCHES: tuple[int, ...] = (1, 2, 4, 8)

    def __init__(
        self,
        simulation_service: SimulationService,
        connectome_path: Path | None = None,
        checkpoints_dir: Path | None = None,
    ) -> None:
        self._simulation_service = simulation_service
        self._connectome_path = connectome_path or ConnectomeConfig.output_path
        self._checkpoints_dir = checkpoints_dir or PPOConfig.checkpoints_dir

        self._lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._broadcast_callback: Callable[[dict[str, Any]], None] | None = None

        self._status: str = "idle"
        self._active_config: dict[str, Any] | None = None
        self._latest_metrics: dict[str, float] | None = None
        self._history: list[dict[str, float]] = []
        self._error_message: str | None = None
        self._saved_checkpoint_file: str | None = None
        self._saved_agent_key: str | None = None
        self._cancel_event: threading.Event | None = None
        self._worker_thread: threading.Thread | None = None

    def bind_event_loop(
        self,
        loop: asyncio.AbstractEventLoop,
        broadcast_callback: Callable[[dict[str, Any]], None],
    ) -> None:
        """Bind server asyncio loop and WebSocket broadcast coroutine scheduler."""
        self._loop = loop
        self._broadcast_callback = broadcast_callback

    def is_training(self) -> bool:
        """Return True while a training job is actively executing or stopping."""
        with self._lock:
            return self._status in ("running", "stopping")

    def get_state(self) -> dict[str, Any]:
        """Return thread-safe snapshot of current or most recent training run."""
        with self._lock:
            return {
                "type": "training_state",
                "status": self._status,
                "config": dict(self._active_config) if self._active_config else None,
                "latest_metrics": (
                    dict(self._latest_metrics) if self._latest_metrics else None
                ),
                "history": list(self._history),
                "error_message": self._error_message,
                "saved_checkpoint_file": self._saved_checkpoint_file,
                "saved_agent_key": self._saved_agent_key,
            }

    def start_job(
        self,
        model_type: str,
        total_timesteps: int = 40960,
        learning_rate: float = 8e-4,
        seed: int = 42,
        save_checkpoint: bool = True,
        save_mode: str | None = None,
        custom_filename: str | None = None,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_coef: float = 0.20,
        ent_coef: float = 0.005,
        vf_coef: float = 0.25,
        update_epochs: int = 4,
        rollout_steps: int = 128,
        sequence_chunk_len: int = 16,
        num_minibatches: int = 4,
        anneal_lr: bool = True,
        save_on_cancel: bool = True,
        resume_checkpoint: str | None = None,
    ) -> dict[str, Any]:
        """Launch a non-blocking CPU PPO training job on a background thread."""
        normalized_model = model_type.lower().strip()
        if normalized_model not in ("fly", "fly_shuffled", "mlp", "gru"):
            raise ValueError(
                f"Unsupported model_type '{model_type}'. Expected fly, fly_shuffled, mlp, or gru."
            )

        if save_mode is not None:
            if save_mode not in self.VALID_SAVE_MODES:
                raise ValueError(
                    f"Unsupported save_mode '{save_mode}'. "
                    f"Expected one of {self.VALID_SAVE_MODES}."
                )
            effective_save_mode = save_mode
        else:
            effective_save_mode = "overwrite_default" if save_checkpoint else "none"

        parsed_total_timesteps = self._parse_int_in_range(
            total_timesteps,
            "total_timesteps",
            1024,
            2_000_000,
        )
        parsed_learning_rate = self._parse_float_in_range(
            learning_rate,
            "learning_rate",
            1e-5,
            1e-2,
        )
        parsed_seed = self._parse_int_in_range(seed, "seed", 0, None)
        parsed_gamma = self._parse_float_in_range(gamma, "gamma", 0.80, 0.999)
        parsed_gae_lambda = self._parse_float_in_range(
            gae_lambda,
            "gae_lambda",
            0.50,
            0.99,
        )
        parsed_clip_coef = self._parse_float_in_range(
            clip_coef,
            "clip_coef",
            0.05,
            0.50,
        )
        parsed_ent_coef = self._parse_float_in_range(ent_coef, "ent_coef", 0.0, 0.10)
        parsed_vf_coef = self._parse_float_in_range(vf_coef, "vf_coef", 0.05, 2.0)
        parsed_update_epochs = self._parse_int_in_range(
            update_epochs,
            "update_epochs",
            1,
            10,
        )
        parsed_rollout_steps = self._parse_int_choice(
            rollout_steps,
            "rollout_steps",
            self.VALID_ROLLOUT_STEPS,
        )
        parsed_chunk_len = self._parse_int_choice(
            sequence_chunk_len,
            "sequence_chunk_len",
            self.VALID_SEQUENCE_CHUNK_LENS,
        )
        parsed_minibatches = self._parse_int_choice(
            num_minibatches,
            "num_minibatches",
            self.VALID_NUM_MINIBATCHES,
        )
        parsed_anneal_lr = self._parse_bool(anneal_lr, "anneal_lr")
        parsed_save_on_cancel = self._parse_bool(save_on_cancel, "save_on_cancel")

        resolved_resume_name: str | None = None
        resume_from_timesteps = 0
        if resume_checkpoint is not None and str(resume_checkpoint).strip() != "":
            candidate_name = Path(str(resume_checkpoint).strip().replace("\\", "/")).name
            resolved_dir = self._checkpoints_dir.resolve()
            candidate_path = (self._checkpoints_dir / candidate_name).resolve()
            if candidate_path.parent != resolved_dir or not candidate_path.is_file():
                raise ValueError(
                    f"Resume checkpoint '{candidate_name}' not found in checkpoints directory."
                )
            resume_payload = torch.load(candidate_path, map_location="cpu", weights_only=True)
            ckpt_model_type = str(resume_payload.get("model_type", ""))
            if ckpt_model_type != normalized_model:
                raise ValueError(
                    f"Resume checkpoint '{candidate_name}' has model_type '{ckpt_model_type}', "
                    f"expected '{normalized_model}'."
                )
            prev_hist = resume_payload.get("history")
            fallback_steps = (
                int(prev_hist["timesteps"][-1])
                if isinstance(prev_hist, dict) and prev_hist.get("timesteps")
                else 0
            )
            resume_from_timesteps = int(
                resume_payload.get("trained_timesteps") or fallback_steps
            )
            if parsed_total_timesteps <= resume_from_timesteps:
                raise ValueError(
                    f"Target total_timesteps ({parsed_total_timesteps}) must be greater than "
                    f"checkpoint trained_timesteps ({resume_from_timesteps})."
                )
            resolved_resume_name = candidate_name

        if effective_save_mode == "overwrite_resumed" and resolved_resume_name is None:
            raise ValueError(
                "save_mode 'overwrite_resumed' requires selecting a resume_checkpoint."
            )

        if parsed_rollout_steps % parsed_chunk_len != 0:
            raise ValueError(
                f"rollout_steps ({parsed_rollout_steps}) must be divisible by "
                f"sequence_chunk_len ({parsed_chunk_len})."
            )
        total_chunks = (parsed_rollout_steps // parsed_chunk_len) * PPOConfig.num_envs
        if total_chunks <= 0 or total_chunks % parsed_minibatches != 0:
            raise ValueError(
                f"Total sequence chunks ({total_chunks}) must be divisible by "
                f"num_minibatches ({parsed_minibatches})."
            )

        with self._lock:
            if self._status in ("running", "stopping"):
                raise RuntimeError("A PPO training job is already in progress.")

            target_filename = self._resolve_target_filename_unlocked(
                normalized_model=normalized_model,
                save_mode=effective_save_mode,
                custom_filename=custom_filename,
                resume_checkpoint=resolved_resume_name,
            )

            self._status = "running"
            self._error_message = None
            self._latest_metrics = None
            self._history = []
            self._saved_checkpoint_file = None
            self._saved_agent_key = None
            self._cancel_event = threading.Event()
            self._active_config = {
                "model_type": normalized_model,
                "total_timesteps": parsed_total_timesteps,
                "resume_checkpoint": resolved_resume_name,
                "resume_from_timesteps": resume_from_timesteps,
                "remaining_timesteps": parsed_total_timesteps - resume_from_timesteps,
                "learning_rate": parsed_learning_rate,
                "seed": parsed_seed,
                "gamma": parsed_gamma,
                "gae_lambda": parsed_gae_lambda,
                "clip_coef": parsed_clip_coef,
                "ent_coef": parsed_ent_coef,
                "vf_coef": parsed_vf_coef,
                "update_epochs": parsed_update_epochs,
                "rollout_steps": parsed_rollout_steps,
                "sequence_chunk_len": parsed_chunk_len,
                "num_minibatches": parsed_minibatches,
                "anneal_lr": parsed_anneal_lr,
                "save_on_cancel": parsed_save_on_cancel,
                "save_checkpoint": bool(effective_save_mode != "none"),
                "save_mode": effective_save_mode,
                "target_checkpoint_file": target_filename,
                "saved_checkpoint_file": None,
                "saved_agent_key": None,
            }

            cancel_event = self._cancel_event
            job_config = dict(self._active_config)

        worker = threading.Thread(
            target=self._run_worker,
            args=(job_config, cancel_event),
            daemon=True,
        )
        with self._lock:
            self._worker_thread = worker

        self._emit_event({**self.get_state(), "type": "training_started"})
        worker.start()
        return self.get_state()

    def stop_job(self) -> dict[str, Any]:
        """Signal the active training job to stop cleanly at the current update boundary."""
        with self._lock:
            if self._status == "running" and self._cancel_event is not None:
                self._status = "stopping"
                self._cancel_event.set()
        return self.get_state()

    def _resolve_target_filename_unlocked(
        self,
        normalized_model: str,
        save_mode: str,
        custom_filename: str | None,
        resume_checkpoint: str | None = None,
    ) -> str | None:
        """Compute safe checkpoint filename inside checkpoints_dir for the requested mode."""
        if save_mode == "none":
            return None
        if save_mode == "overwrite_default":
            return f"{normalized_model}_ppo.pt"
        if save_mode == "overwrite_resumed" and resume_checkpoint:
            return resume_checkpoint

        raw_name = str(custom_filename or "").strip()
        normalized_slashes = raw_name.replace("\\", "/")
        basename = Path(normalized_slashes).name
        raw_stem = "" if basename.lower() == ".pt" else Path(basename).stem
        sanitized_stem = re.sub(r"[^A-Za-z0-9_-]", "_", raw_stem).strip("_")
        base_stem = (
            sanitized_stem
            or f"{normalized_model}_ppo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

        candidate_name = f"{base_stem}.pt"
        suffix_index = 2
        while (self._checkpoints_dir / candidate_name).exists():
            candidate_name = f"{base_stem}_{suffix_index}.pt"
            suffix_index += 1

        resolved_dir = self._checkpoints_dir.resolve()
        resolved_target = (self._checkpoints_dir / candidate_name).resolve()
        if resolved_target.parent != resolved_dir:
            raise ValueError(
                f"Resolved checkpoint path escapes checkpoints directory: {candidate_name}"
            )
        return candidate_name

    def _run_worker(
        self,
        job_config: dict[str, Any],
        cancel_event: threading.Event,
    ) -> None:
        model_type = str(job_config["model_type"])
        effective_save_mode = str(job_config["save_mode"])
        target_filename: str | None = job_config["target_checkpoint_file"]

        try:
            artifact: ConnectomeArtifact | None = None
            resolved_conn_path = self._connectome_path
            if model_type == "fly":
                artifact = ConnectomeArtifact.load(self._connectome_path)
            elif model_type == "fly_shuffled":
                resolved_conn_path = default_shuffled_connectome_path(self._connectome_path)
                artifact = load_or_create_shuffled_connectome(
                    connectome_path=self._connectome_path,
                    shuffled_path=resolved_conn_path,
                )

            ppo_config = PPOConfig(
                total_timesteps=int(job_config["total_timesteps"]),
                rollout_steps=int(job_config["rollout_steps"]),
                sequence_chunk_len=int(job_config["sequence_chunk_len"]),
                num_minibatches=int(job_config["num_minibatches"]),
                update_epochs=int(job_config["update_epochs"]),
                learning_rate=float(job_config["learning_rate"]),
                gamma=float(job_config["gamma"]),
                gae_lambda=float(job_config["gae_lambda"]),
                clip_coef=float(job_config["clip_coef"]),
                ent_coef=float(job_config["ent_coef"]),
                vf_coef=float(job_config["vf_coef"]),
                anneal_lr=bool(job_config["anneal_lr"]),
                seed=int(job_config["seed"]),
                checkpoints_dir=self._checkpoints_dir,
            )
            save_on_cancel = bool(job_config.get("save_on_cancel", True))
            trainer = PPOTrainer(
                ppo_config=ppo_config,
                env_config=TrafficEnvConfig(),
                connectome_config=ConnectomeConfig(output_path=resolved_conn_path),
            )
            preview_env = TrafficIntersectionEnv(
                config=TrafficEnvConfig(profile_name="rush_hour")
            )
            preview_obs, _ = preview_env.reset(seed=int(job_config["seed"]))
            preview_hidden: torch.Tensor | None = None

            def _on_policy_update(
                policy: BaseRecurrentActorCritic,
                metrics: dict[str, float],
            ) -> None:
                nonlocal preview_obs, preview_hidden
                with self._lock:
                    self._latest_metrics = dict(metrics)
                    self._history.append(dict(metrics))

                preview_frames, preview_obs, preview_hidden = self._rollout_preview_frames(
                    policy=policy,
                    preview_env=preview_env,
                    preview_obs=preview_obs,
                    preview_hidden=preview_hidden,
                )
                self._emit_event(
                    {
                        "type": "training_update",
                        "status": "running",
                        "config": job_config,
                        "metrics": metrics,
                        "preview_frames": preview_frames,
                    }
                )

            resume_ckpt_name = job_config.get("resume_checkpoint")
            resume_from_path = (
                self._checkpoints_dir / str(resume_ckpt_name)
                if resume_ckpt_name
                else None
            )

            try:
                trainer.train(
                    model_type=model_type,
                    artifact=artifact,
                    verbose=False,
                    on_policy_update_callback=_on_policy_update,
                    cancel_event=cancel_event,
                    save_checkpoint=(effective_save_mode != "none"),
                    checkpoint_filename=target_filename,
                    save_on_cancel=save_on_cancel,
                    resume_from=resume_from_path,
                )
            finally:
                preview_env.close()

            if cancel_event.is_set():
                cancelled_checkpoint_file: str | None = None
                cancelled_agent_key: str | None = None
                with self._lock:
                    has_updates = len(self._history) > 0
                if save_on_cancel and target_filename is not None and has_updates:
                    cancelled_agent_key, _ = self._simulation_service.register_checkpoint(
                        self._checkpoints_dir / target_filename,
                        activate=True,
                    )
                    cancelled_checkpoint_file = target_filename

                with self._lock:
                    self._saved_checkpoint_file = cancelled_checkpoint_file
                    self._saved_agent_key = cancelled_agent_key
                    if self._active_config is not None:
                        self._active_config["saved_checkpoint_file"] = (
                            cancelled_checkpoint_file
                        )
                        self._active_config["saved_agent_key"] = cancelled_agent_key
                    self._status = "cancelled"
                self._emit_event({**self.get_state(), "type": "training_cancelled"})
                return

            saved_checkpoint_file: str | None = None
            saved_agent_key: str | None = None
            if target_filename is not None:
                saved_agent_key, _ = self._simulation_service.register_checkpoint(
                    self._checkpoints_dir / target_filename,
                    activate=True,
                )
                saved_checkpoint_file = target_filename

            with self._lock:
                self._saved_checkpoint_file = saved_checkpoint_file
                self._saved_agent_key = saved_agent_key
                if self._active_config is not None:
                    self._active_config["saved_checkpoint_file"] = saved_checkpoint_file
                    self._active_config["saved_agent_key"] = saved_agent_key
                self._status = "completed"
            self._emit_event({**self.get_state(), "type": "training_completed"})
        except Exception as exc:
            with self._lock:
                self._status = "error"
                self._error_message = str(exc)
            self._emit_event({**self.get_state(), "type": "training_error"})

    @staticmethod
    def _rollout_preview_frames(
        policy: BaseRecurrentActorCritic,
        preview_env: TrafficIntersectionEnv,
        preview_obs: np.ndarray,
        preview_hidden: torch.Tensor | None,
        num_ticks: int = 4,
    ) -> tuple[list[dict[str, Any]], np.ndarray, torch.Tensor]:
        """Step the preview environment for live training inspection frames."""
        policy.eval()
        current_obs = preview_obs
        current_hidden = (
            policy.initial_hidden(batch_size=1, device="cpu")
            if preview_hidden is None
            else preview_hidden
        )
        preview_frames: list[dict[str, Any]] = []

        with torch.no_grad():
            for _ in range(num_ticks):
                obs_tensor = torch.as_tensor(
                    current_obs,
                    dtype=torch.float32,
                    device="cpu",
                ).unsqueeze(0)
                logits, _value, current_hidden = policy.forward_step(
                    obs_tensor,
                    current_hidden,
                )
                chosen_action = int(torch.argmax(logits, dim=-1).item())
                current_obs, _reward, terminated, truncated, _info = preview_env.step(
                    chosen_action
                )
                preview_frames.append(preview_env.get_render_snapshot())

                if terminated or truncated:
                    current_obs, _ = preview_env.reset()
                    current_hidden = policy.initial_hidden(batch_size=1, device="cpu")

        return preview_frames, current_obs, current_hidden

    def _emit_event(self, payload: dict[str, Any]) -> None:
        if self._loop is not None and self._broadcast_callback is not None:
            if not self._loop.is_closed():
                self._loop.call_soon_threadsafe(self._broadcast_callback, payload)

    @staticmethod
    def _parse_int_in_range(
        value: Any,
        name: str,
        min_value: int,
        max_value: int | None,
    ) -> int:
        if isinstance(value, bool) or (
            isinstance(value, float) and not value.is_integer()
        ):
            raise ValueError(f"{name} must be an integer.")
        parsed = int(value)
        if parsed < min_value or (max_value is not None and parsed > max_value):
            upper = str(max_value) if max_value is not None else "infinity"
            raise ValueError(f"{name} must be in [{min_value}, {upper}], got {parsed}.")
        return parsed

    @staticmethod
    def _parse_float_in_range(
        value: Any,
        name: str,
        min_value: float,
        max_value: float,
    ) -> float:
        if isinstance(value, bool):
            raise ValueError(f"{name} must be a finite float.")
        parsed = float(value)
        if not math.isfinite(parsed) or parsed < min_value or parsed > max_value:
            raise ValueError(
                f"{name} must be in [{min_value}, {max_value}], got {parsed}."
            )
        return parsed

    @classmethod
    def _parse_int_choice(
        cls,
        value: Any,
        name: str,
        allowed: tuple[int, ...],
    ) -> int:
        parsed = cls._parse_int_in_range(value, name, min(allowed), max(allowed))
        if parsed not in allowed:
            raise ValueError(f"{name} must be one of {allowed}, got {parsed}.")
        return parsed

    @staticmethod
    def _parse_bool(value: Any, name: str) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered == "true":
                return True
            if lowered == "false":
                return False
        raise ValueError(f"{name} must be a boolean.")
