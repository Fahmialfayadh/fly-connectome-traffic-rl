"""Centralized configuration dataclasses for all project modules."""

from dataclasses import dataclass, field
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_PATH = PROJECT_ROOT / "dataset" / "connections_princeton.csv"
DEFAULT_ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"


@dataclass(frozen=True)
class ConnectomeConfig:
    """Configuration for extracting and scaling the Drosophila sub-connectome."""

    dataset_path: Path = DEFAULT_DATASET_PATH
    output_path: Path = DEFAULT_ARTIFACTS_DIR / "fly_subconnectome.pt"
    cx_neuropils: tuple[str, ...] = ("PB", "FB", "EB", "NO")
    mb_neuropils: tuple[str, ...] = (
        "MB_CA_L",
        "MB_CA_R",
        "MB_ML_L",
        "MB_ML_R",
        "MB_PED_L",
        "MB_PED_R",
        "MB_VL_L",
        "MB_VL_R",
    )
    cx_neuron_quota: int = 1800
    mb_neuron_quota: int = 700
    excitatory_nts: tuple[str, ...] = ("ACH",)
    inhibitory_nts: tuple[str, ...] = ("GABA", "GLUT")
    target_spectral_radius: float = 0.95
    power_iteration_seed: int = 42


@dataclass(frozen=True)
class TrafficEnvConfig:
    """Configuration for the 4-way, 2-lane Gymnasium traffic environment."""

    dt_seconds: float = 2.0
    max_steps: int = 500
    max_lane_capacity: int = 30
    queue_obs_norm_capacity: float = 30.0
    hard_max_lane_capacity: int = 80
    straight_discharge_rate: float = 2.5
    turn_discharge_rate: float = 2.0
    min_green_steps: int = 6
    yellow_steps: int = 2
    all_red_steps: int = 1
    wait_penalty_alpha: float = 0.1
    switch_penalty_beta: float = 1.0
    reward_scale: float = 0.005
    max_wait_norm_seconds: float = 150.0
    max_elapsed_norm_steps: float = 20.0
    profile_name: str = "mixed"
    collision_penalty: float = 80.0
    red_light_wait_threshold_s: float = 150.0
    red_light_max_prob: float = 0.14
    dilemma_zone_prob: float = 0.18
    enable_collisions: bool = True


@dataclass(frozen=True)
class PPOConfig:
    """Hyperparameters for recurrent and feedforward CPU PPO training."""

    total_timesteps: int = 163840
    num_envs: int = 8
    rollout_steps: int = 128
    sequence_chunk_len: int = 16
    num_minibatches: int = 4
    update_epochs: int = 4
    learning_rate: float = 8e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.005
    vf_coef: float = 0.25
    max_grad_norm: float = 0.5
    anneal_lr: bool = True
    seed: int = 42
    num_cpu_threads: int = 4
    checkpoints_dir: Path = DEFAULT_ARTIFACTS_DIR / "checkpoints"


@dataclass(frozen=True)
class BenchmarkConfig:
    """Configuration for multi-scenario evaluation and pharmacology sweeps."""

    scenarios: tuple[str, ...] = ("normal", "asymmetric", "rush_hour", "stress_test")
    eval_seeds: tuple[int, ...] = field(
        default_factory=lambda: tuple(range(1000, 1010))
    )
    latency_warmup_steps: int = 20
    latency_measure_steps: int = 200
    output_dir: Path = DEFAULT_ARTIFACTS_DIR / "benchmarks"


@dataclass(frozen=True)
class SandboxUIConfig:
    """Configuration for the interactive 60 FPS Pygame sandbox window."""

    window_width: int = 1280
    window_height: int = 760
    left_panel_width: int = 680
    right_panel_width: int = 600
    target_fps: int = 60
    sim_steps_per_second: float = 5.0
    heatmap_display_size: int = 360
    heatmap_grid_side: int = 50
    gain_min: float = 0.0
    gain_max: float = 2.5
    lesion_fraction: float = 0.25
    window_title: str = "Drosophila Connectome Traffic Control Sandbox"
