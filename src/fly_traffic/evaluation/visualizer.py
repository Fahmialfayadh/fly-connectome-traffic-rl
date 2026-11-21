"""Matplotlib benchmark visualizations with neutral factual titles and labels."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fly_traffic.evaluation.evaluator import PharmacologyResult, ScenarioBenchmarkResult


class BenchmarkPlotter:
    """Generates training curve, scenario comparison, and pharmacology bar charts."""

    AGENT_COLORS: dict[str, str] = {
        "Fixed Timer": "#6c757d",
        "PPO-MLP": "#2b6cb0",
        "PPO-GRU": "#dd6b20",
        "Shuffled Connectome PPO": "#d69e2e",
        "Fly Connectome PPO": "#2f855a",
    }

    @classmethod
    def plot_training_curves(
        cls,
        histories: dict[str, dict[str, list[float]]],
        output_path: Path,
    ) -> Path:
        """Plot mean_episode_reward, mean_throughput, and mean_collisions vs timesteps."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8), dpi=150)

        for label, hist in histories.items():
            steps = np.array(hist.get("timesteps", []), dtype=np.float64)
            rewards = np.array(hist.get("mean_episode_reward", []), dtype=np.float64)
            throughputs = np.array(hist.get("mean_throughput", []), dtype=np.float64)
            collisions = np.array(hist.get("mean_collisions", []), dtype=np.float64)
            if len(steps) == 0:
                continue

            color = cls.AGENT_COLORS.get(label, None)
            axes[0].plot(steps, rewards, label=label, color=color, linewidth=2.0)
            axes[1].plot(steps, throughputs, label=label, color=color, linewidth=2.0)
            if len(collisions) == len(steps):
                axes[2].plot(steps, collisions, label=label, color=color, linewidth=2.0)

        axes[0].set_title("mean_episode_reward vs timesteps")
        axes[0].set_xlabel("timesteps")
        axes[0].set_ylabel("mean_episode_reward")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        axes[1].set_title("mean_throughput vs timesteps")
        axes[1].set_xlabel("timesteps")
        axes[1].set_ylabel("mean_throughput (vehicles)")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()

        axes[2].set_title("mean_collisions vs timesteps")
        axes[2].set_xlabel("timesteps")
        axes[2].set_ylabel("mean_collisions (per episode)")
        axes[2].grid(True, alpha=0.3)
        if axes[2].lines:
            axes[2].legend()

        fig.tight_layout()
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        return output_path

    @classmethod
    def plot_scenario_comparison(
        cls,
        results: list[ScenarioBenchmarkResult],
        output_path: Path,
    ) -> Path:
        """Plot grouped bar charts for reward, wait time, throughput, and collisions."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        scenarios = list(dict.fromkeys(r.scenario for r in results))
        agents = list(dict.fromkeys(r.agent_name for r in results))
        lookup = {(r.scenario, r.agent_name): r for r in results}

        fig, axes_grid = plt.subplots(2, 2, figsize=(13.5, 9.0), dpi=150)
        axes = axes_grid.ravel()
        metrics: list[tuple[str, str | None, str]] = [
            ("mean_reward", "std_reward", "mean_reward"),
            ("mean_wait_time_s", "std_wait_time_s", "mean_wait_time_s (s)"),
            ("mean_throughput", "std_throughput", "mean_throughput (vehicles)"),
            ("mean_collisions", None, "mean_collisions (per episode)"),
        ]

        x = np.arange(len(scenarios))
        bar_width = 0.8 / max(1, len(agents))

        for ax, (metric_attr, std_attr, ylabel) in zip(axes, metrics):
            for idx, agent_name in enumerate(agents):
                offsets = x + (idx - (len(agents) - 1) / 2.0) * bar_width
                means = [
                    getattr(lookup[(sc, agent_name)], metric_attr)
                    for sc in scenarios
                ]
                stds = (
                    [getattr(lookup[(sc, agent_name)], std_attr) for sc in scenarios]
                    if std_attr is not None
                    else None
                )
                color = cls.AGENT_COLORS.get(agent_name, None)
                ax.bar(
                    offsets,
                    means,
                    width=bar_width,
                    yerr=stds,
                    capsize=3 if stds is not None else 0,
                    label=agent_name,
                    color=color,
                    alpha=0.9,
                )

            ax.set_title(metric_attr)
            ax.set_xticks(x)
            ax.set_xticklabels(scenarios)
            ax.set_xlabel("scenario")
            ax.set_ylabel(ylabel)
            ax.grid(True, axis="y", alpha=0.3)

        axes[0].legend(loc="best", fontsize=8)
        fig.tight_layout()
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        return output_path

    @classmethod
    def plot_pharmacology_ablation(
        cls,
        results: list[PharmacologyResult],
        output_path: Path,
    ) -> Path:
        """Plot mean_reward, mean_wait_time_s, and mean_collisions across pharmacology/lesion runs."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        labels = [r.condition for r in results]
        rewards = [r.mean_reward for r in results]
        waits = [r.mean_wait_time_s for r in results]
        collisions = [r.mean_collisions for r in results]

        y = np.arange(len(labels))
        fig, axes = plt.subplots(1, 3, figsize=(17.0, 5.2), dpi=150)

        axes[0].barh(y, rewards, color="#2f855a", alpha=0.9)
        axes[0].set_yticks(y)
        axes[0].set_yticklabels(labels, fontsize=9)
        axes[0].invert_yaxis()
        axes[0].set_title("mean_reward by condition (rush_hour)")
        axes[0].set_xlabel("mean_reward")
        axes[0].grid(True, axis="x", alpha=0.3)

        axes[1].barh(y, waits, color="#b7791f", alpha=0.9)
        axes[1].set_yticks(y)
        axes[1].set_yticklabels(labels, fontsize=9)
        axes[1].invert_yaxis()
        axes[1].set_title("mean_wait_time_s by condition (rush_hour)")
        axes[1].set_xlabel("mean_wait_time_s (s)")
        axes[1].grid(True, axis="x", alpha=0.3)

        axes[2].barh(y, collisions, color="#c53030", alpha=0.9)
        axes[2].set_yticks(y)
        axes[2].set_yticklabels(labels, fontsize=9)
        axes[2].invert_yaxis()
        axes[2].set_title("mean_collisions by condition (rush_hour)")
        axes[2].set_xlabel("mean_collisions (per episode)")
        axes[2].grid(True, axis="x", alpha=0.3)

        fig.tight_layout()
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        return output_path
