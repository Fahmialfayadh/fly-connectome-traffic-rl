"""CLI entrypoint to train Fly Connectome, MLP, and GRU PPO agents on CPU."""

import argparse
from pathlib import Path

from fly_traffic.agents.baselines import build_policy
from fly_traffic.config import ConnectomeConfig, PPOConfig, TrafficEnvConfig
from fly_traffic.connectome.artifact import ConnectomeArtifact
from fly_traffic.training.ppo_trainer import PPOTrainer


def parse_args() -> argparse.Namespace:
    """Parse CLI training options."""
    parser = argparse.ArgumentParser(
        description="Train Drosophila Connectome PPO and baseline policies on CPU."
    )
    parser.add_argument(
        "--agents",
        nargs="+",
        default=["fly", "mlp", "gru"],
        choices=["fly", "mlp", "gru"],
        help="Policies to train sequentially.",
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
        help="Number of synchronous parallel CPU environments.",
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
    return parser.parse_args()


def print_parameter_table(
    agents: list[str],
    artifact: ConnectomeArtifact | None,
) -> None:
    """Print parameter counts across architectures to verify fair baseline sizing."""
    print("=== Policy Parameter Comparison (obs_dim=22, act_dim=4) ===")
    for model_type in agents:
        policy = build_policy(model_type, obs_dim=22, act_dim=4, artifact=artifact)
        trainable = policy.count_trainable_parameters()
        total = policy.count_total_parameters()
        print(
            f"  {model_type.upper():4s} | trainable={trainable:7d} | "
            f"total_params={total:7d} | hidden_dim={policy.hidden_dim}"
        )


def main() -> None:
    """Train requested RL policies and persist checkpoints."""
    args = parse_args()
    conn_config = ConnectomeConfig(output_path=args.connectome)
    env_config = TrafficEnvConfig()
    ppo_config = PPOConfig(
        total_timesteps=args.total_timesteps,
        num_envs=args.num_envs,
        seed=args.seed,
        checkpoints_dir=args.checkpoints_dir,
    )

    artifact: ConnectomeArtifact | None = None
    if "fly" in args.agents:
        if not conn_config.output_path.exists():
            raise FileNotFoundError(
                f"Sub-connectome artifact not found at {conn_config.output_path}. "
                "Run scripts/extract_connectome.py first."
            )
        artifact = ConnectomeArtifact.load(conn_config.output_path)

    print_parameter_table(args.agents, artifact)

    trainer = PPOTrainer(
        ppo_config=ppo_config,
        env_config=env_config,
        connectome_config=conn_config,
    )

    for model_type in args.agents:
        print(f"\n--- Training {model_type.upper()} PPO ({args.total_timesteps} steps) ---")
        trainer.train(model_type=model_type, artifact=artifact, verbose=True)


if __name__ == "__main__":
    main()
