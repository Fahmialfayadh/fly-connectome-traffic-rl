"""CLI entrypoint to benchmark trained agents and generate quantitative figures."""

import argparse
from pathlib import Path

from fly_traffic.agents.base import BaseTrafficAgent, NeuralAgentAdapter
from fly_traffic.agents.baselines import load_policy_checkpoint
from fly_traffic.agents.fixed_timer import FixedTimerAgent
from fly_traffic.agents.fly_policy import FlyConnectomePolicy
from fly_traffic.config import BenchmarkConfig, ConnectomeConfig, PPOConfig, TrafficEnvConfig
from fly_traffic.connectome.artifact import ConnectomeArtifact
from fly_traffic.connectome.shuffled import (
    default_shuffled_connectome_path,
    load_or_create_shuffled_connectome,
)
from fly_traffic.evaluation.evaluator import AgentEvaluator
from fly_traffic.evaluation.visualizer import BenchmarkPlotter


def parse_args() -> argparse.Namespace:
    """Parse CLI options for evaluation and plotting."""
    parser = argparse.ArgumentParser(
        description=(
            "Run benchmark comparison across Fly Connectome PPO, Shuffled Connectome PPO, "
            "PPO-GRU, PPO-MLP, and Fixed Timer."
        )
    )
    parser.add_argument(
        "--checkpoints-dir",
        type=Path,
        default=PPOConfig.checkpoints_dir,
        help="Directory containing trained *_ppo.pt checkpoints.",
    )
    parser.add_argument(
        "--connectome",
        type=Path,
        default=ConnectomeConfig.output_path,
        help="Path to fly_subconnectome.pt.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=BenchmarkConfig.output_dir,
        help="Directory for JSON, Markdown table, and PNG plots.",
    )
    return parser.parse_args()


def main() -> None:
    """Load trained checkpoints, run benchmarks, and render factual figures."""
    args = parse_args()
    artifact = (
        ConnectomeArtifact.load(args.connectome)
        if args.connectome.exists()
        else None
    )
    shuffled_artifact = (
        load_or_create_shuffled_connectome(
            connectome_path=args.connectome,
            shuffled_path=default_shuffled_connectome_path(args.connectome),
        )
        if args.connectome.exists()
        else None
    )

    agents: list[BaseTrafficAgent] = []
    histories: dict[str, dict[str, list[float]]] = {}
    fly_policy: FlyConnectomePolicy | None = None

    checkpoint_specs = [
        ("fly", "Fly Connectome PPO", args.checkpoints_dir / "fly_ppo.pt"),
        ("fly_shuffled", "Shuffled Connectome PPO", args.checkpoints_dir / "fly_shuffled_ppo.pt"),
        ("gru", "PPO-GRU", args.checkpoints_dir / "gru_ppo.pt"),
        ("mlp", "PPO-MLP", args.checkpoints_dir / "mlp_ppo.pt"),
    ]

    for model_key, display_name, ckpt_path in checkpoint_specs:
        if not ckpt_path.exists():
            print(f"Skipping missing checkpoint: {ckpt_path}")
            continue

        policy, payload = load_policy_checkpoint(
            path=ckpt_path,
            artifact=artifact if model_key == "fly" else None,
            shuffled_artifact=shuffled_artifact if model_key == "fly_shuffled" else None,
        )
        agents.append(NeuralAgentAdapter(policy=policy, agent_name=display_name))
        if isinstance(payload.get("history"), dict):
            histories[display_name] = payload["history"]
        if model_key == "fly" and isinstance(policy, FlyConnectomePolicy):
            fly_policy = policy

    agents.append(FixedTimerAgent())

    evaluator = AgentEvaluator(
        benchmark_config=BenchmarkConfig(output_dir=args.output_dir),
        env_config=TrafficEnvConfig(),
    )

    scenario_results = evaluator.evaluate_all_agents(agents)
    pharmacology_results = (
        evaluator.evaluate_pharmacology_and_lesions(fly_policy)
        if fly_policy is not None
        else []
    )

    json_path, md_path = evaluator.save_reports(
        output_dir=args.output_dir,
        scenario_results=scenario_results,
        pharmacology_results=pharmacology_results,
    )

    if histories:
        BenchmarkPlotter.plot_training_curves(
            histories=histories,
            output_path=args.output_dir / "training_curves.png",
        )
    BenchmarkPlotter.plot_scenario_comparison(
        results=scenario_results,
        output_path=args.output_dir / "scenario_comparison.png",
    )
    BenchmarkPlotter.plot_single_scenario_metric(
        results=scenario_results,
        metric_attr="mean_reward",
        std_attr="std_reward",
        ylabel="mean_reward",
        output_path=args.output_dir / "mean_reward.png",
    )
    BenchmarkPlotter.plot_single_scenario_metric(
        results=scenario_results,
        metric_attr="mean_collisions",
        std_attr=None,
        ylabel="mean_collisions (per episode)",
        output_path=args.output_dir / "mean_collisions.png",
        label_nonzero=True,
    )
    if pharmacology_results:
        BenchmarkPlotter.plot_pharmacology_ablation(
            results=pharmacology_results,
            output_path=args.output_dir / "pharmacology_ablation.png",
        )

    print(f"Saved JSON report: {json_path}")
    print(f"Saved Markdown table: {md_path}")
    print("\n" + md_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
