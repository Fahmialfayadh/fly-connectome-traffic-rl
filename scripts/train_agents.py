
"""CLI entrypoint to train Fly Connectome, MLP, and GRU PPO agents on CPU."""

import argparse
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from fly_traffic.agents.baselines import build_policy
from fly_traffic.config import ConnectomeConfig, PPOConfig, TrafficEnvConfig
from fly_traffic.connectome.artifact import ConnectomeArtifact
from fly_traffic.training.ppo_trainer import PPOTrainer

import shutil
from datetime import datetime


def parse_args() -> argparse.Namespace:
    """Parse CLI training options."""
    parser = argparse.ArgumentParser(
        description="Train Drosophila Connectome PPO and baseline policies on CPU."
    )
    parser.add_argument(
        "--agents",
        nargs="+",
        default=["fly", "mlp", "gru"],
        choices=["fly", "fly_shuffled", "mlp", "gru"],
        help="Policies to train.",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=PPOConfig.total_timesteps,
        help="Total environment steps per trained policy.",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=PPOConfig.num_envs,
        help="Number of synchronous parallel CPU environments per model.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=PPOConfig.seed,
        help="Random seed for reproducible training.",
    )
    parser.add_argument(
        "--connectome",
        type=Path,
        default=ConnectomeConfig.output_path,
        help="Path to extracted fly_subconnectome.pt artifact.",
    )
    parser.add_argument(
        "--checkpoints-dir",
        type=Path,
        default=PPOConfig.checkpoints_dir,
        help="Directory where trained checkpoints are saved.",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="",
        help="Suffix appended to each model checkpoint filename.",
    )
    parser.add_argument(
        "--parallel-models",
        type=int,
        choices=[1, 2, 4],
        default=1,
        help="Maximum number of models trained concurrently.",
    )


    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from an existing checkpoint when available.",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=PPOConfig.learning_rate,
        help="Learning rate for PPO optimizer.",
    )

    parser.add_argument(
        "--copy-checkpoints",
        action="store_true",
        help="Copy checkpoints to a new directory instead of training.",
    )

    return parser.parse_args()


def print_parameter_table(
    agents: list[str],
    artifact: ConnectomeArtifact | None,
) -> None:
    """Print parameter counts across architectures."""
    print("=== Policy Parameter Comparison (obs_dim=22, act_dim=4) ===")

    for model_type in agents:
        policy = build_policy(
            model_type,
            obs_dim=22,
            act_dim=4,
            artifact=artifact,
        )
        trainable = policy.count_trainable_parameters()
        total = policy.count_total_parameters()

        print(
            f"  {model_type.upper():4s} | trainable={trainable:7d} | "
            f"total_params={total:7d} | hidden_dim={policy.hidden_dim}"
        )


def train_agent_worker(
    model_type: str,
    total_timesteps: int,
    num_envs: int,
    seed: int,
    connectome_path: Path,
    checkpoints_dir: Path,
    suffix: str = "",
    resume: bool = False,
    lr: float = PPOConfig.learning_rate,
    ppo_overrides: dict | None = None,
    env_overrides: dict | None = None,
) -> None:
    """Train one independent PPO agent inside a separate process."""

    # Prevent each process from spawning excessive PyTorch CPU threads.
    import torch
    torch.set_num_threads(1)

    conn_config = ConnectomeConfig(output_path=connectome_path)
    env_config = TrafficEnvConfig(**(env_overrides or {}))

    ppo_kwargs = dict(ppo_overrides or {})
    ppo_kwargs.update(
        total_timesteps=total_timesteps,
        num_envs=num_envs,
        seed=seed,
        checkpoints_dir=checkpoints_dir,
        learning_rate=lr,
    )
    ppo_config = PPOConfig(**ppo_kwargs)

    artifact: ConnectomeArtifact | None = None

    if model_type == "fly":
        artifact = ConnectomeArtifact.load(conn_config.output_path)
    elif model_type == "fly_shuffled":
        from fly_traffic.connectome.shuffled import (
            default_shuffled_connectome_path,
            load_or_create_shuffled_connectome,
        )
        shuffled_path = default_shuffled_connectome_path(conn_config.output_path)
        artifact = load_or_create_shuffled_connectome(
            connectome_path=conn_config.output_path,
            shuffled_path=shuffled_path,
        )

    trainer = PPOTrainer(
        ppo_config=ppo_config,
        env_config=env_config,
        connectome_config=conn_config,
    )

    target_filename = f"{model_type}_ppo{suffix}.pt"
    checkpoint_path = checkpoints_dir / target_filename
    resume_from = checkpoint_path if (resume and checkpoint_path.exists()) else None
    mode_label = f"Resuming from {checkpoint_path.name}" if resume_from else "Starting fresh"

    print(
        f"\n[PID {os.getpid()}] {mode_label} for {model_type.upper()} PPO "
        f"(target={total_timesteps} steps)",
        flush=True,
    )

    trainer.train(
        model_type=model_type,
        artifact=artifact,
        verbose=True,
        checkpoint_filename=target_filename,
        resume_from=resume_from,
    )

    print(
        f"[PID {os.getpid()}] Finished {model_type.upper()} PPO",
        flush=True,
    )

def copy_checkpoints(
    agents: list[str],
    checkpoints_dir: Path,
    suffix: str | None,
    backup_dir: Path,
    timestamp: str,
) -> None:
    """Copy existing checkpoints to a new directory with a suffix."""

    for model_type in agents:
        model_key = model_type.lower()
        suffix = suffix or ""
        source_path = checkpoints_dir / f"{model_key}_ppo{suffix}.pt"

        if not source_path.exists():
            print(f"Skipping missing checkpoint: {source_path}")
            continue

        backup_path = backup_dir / f"{model_key}_ppo{suffix}_{timestamp}.pt"
        shutil.copy2(source_path, backup_path)
        print(f"Backed up {source_path} -> {backup_path}")
    return

def main() -> None:
    """Train requested RL policies sequentially or concurrently."""
    args = parse_args()

    conn_config = ConnectomeConfig(output_path=args.connectome)
    artifact: ConnectomeArtifact | None = None

    if "fly" in args.agents or "fly_shuffled" in args.agents:
        if not conn_config.output_path.exists():
            raise FileNotFoundError(
                f"Sub-connectome artifact not found at {conn_config.output_path}. "
                "Run scripts/extract_connectome.py first."
            )

        artifact = ConnectomeArtifact.load(conn_config.output_path)

    print_parameter_table(args.agents, artifact)

    if args.copy_checkpoints:
        if not args.suffix:
            raise ValueError("`--suffix` harus diisi saat menggunakan `--copy-checkpoints`.")
        
        checkpoints_dir = Path(args.checkpoints_dir)
        backup_dir = checkpoints_dir / "backups"
        backup_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        copy_checkpoints(
            agents=args.agents,
            checkpoints_dir=args.checkpoints_dir,
            suffix=args.suffix,
            backup_dir=backup_dir,
            timestamp=timestamp,
        )

    # SEQUENTIAL MODE: preserve the original training behavior.
    if args.parallel_models == 1:
        ppo_config = PPOConfig(
            total_timesteps=args.total_timesteps,
            num_envs=args.num_envs,
            seed=args.seed,
            checkpoints_dir=args.checkpoints_dir,
        )

        trainer = PPOTrainer(
            ppo_config=ppo_config,
            env_config=TrafficEnvConfig(),
            connectome_config=conn_config,
        )

        for model_type in args.agents:
            model_artifact = artifact
            if model_type == "fly_shuffled":
                from fly_traffic.connectome.shuffled import (
                    default_shuffled_connectome_path,
                    load_or_create_shuffled_connectome,
                )
                shuffled_path = default_shuffled_connectome_path(conn_config.output_path)
                model_artifact = load_or_create_shuffled_connectome(
                    connectome_path=conn_config.output_path,
                    shuffled_path=shuffled_path,
                )

            target_filename = f"{model_type}_ppo{args.suffix}.pt"
            checkpoint_path = args.checkpoints_dir / target_filename
            resume_from = (
                checkpoint_path if (args.resume and checkpoint_path.exists()) else None
            )
            if resume_from is not None:
                print(
                    f"\n--- Resuming {model_type.upper()} PPO from {checkpoint_path} "
                    f"(target={args.total_timesteps} steps) ---"
                )
            else:
                if args.resume:
                    print(
                        f"\n--- No existing checkpoint at {checkpoint_path}; "
                        f"starting {model_type.upper()} PPO fresh ({args.total_timesteps} steps) ---"
                    )
                else:
                    print(
                        f"\n--- Training {model_type.upper()} PPO "
                        f"({args.total_timesteps} steps) ---"
                    )
            print(f"learning rate {model_type}:", args.lr)

            trainer.train(
                model_type=model_type,
                artifact=model_artifact,
                verbose=True,
                checkpoint_filename=target_filename,
                resume_from=resume_from,
                learning_rate=args.lr,
            )

        return

    # PARALLEL MODE: independent process for each training task.
    max_workers = min(args.parallel_models, len(args.agents))

    print("\n=== PARALLEL PPO TRAINING ===")
    print(f"Requested agents : {args.agents}")
    print(f"Concurrent models: {max_workers}")
    print(f"Timesteps/model  : {args.total_timesteps}")
    print(f"Environments/model: {args.num_envs}")
    print(f"Resume enabled   : {args.resume}")

    with ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=mp.get_context("spawn"),
    ) as executor:

        futures = {
            executor.submit(
                train_agent_worker,
                model_type,
                args.total_timesteps,
                args.num_envs,
                args.seed,
                args.connectome,
                args.checkpoints_dir,
                args.suffix,
                args.resume,
                args.lr,
            ): model_type
            for model_type in args.agents
        }

        for future in as_completed(futures):
            model_type = futures[future]

            try:
                future.result()
                print(f"[MAIN] {model_type.upper()} completed successfully.")

            except Exception as exc:
                raise RuntimeError(
                    f"{model_type.upper()} PPO training failed."
                ) from exc

    print("\n=== ALL PPO TRAINING COMPLETED ===")


if __name__ == "__main__":
    main()
