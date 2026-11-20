"""Multi-scenario evaluation, CPU latency profiling, and pharmacology/ablation suite."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time

import numpy as np

from fly_traffic.agents.base import BaseTrafficAgent, NeuralAgentAdapter
from fly_traffic.agents.fly_policy import FlyConnectomePolicy
from fly_traffic.config import BenchmarkConfig, TrafficEnvConfig
from fly_traffic.environment.traffic_env import TrafficIntersectionEnv


@dataclass(frozen=True)
class EpisodeMetrics:
    """Factual episode summary metrics measured over one seeded evaluation run."""

    scenario: str
    seed: int
    agent_name: str
    total_reward: float
    mean_queue: float
    mean_wait_time_s: float
    max_wait_time_s: float
    total_throughput: int
    total_switches: int
    total_collisions: int = 0


@dataclass(frozen=True)
class ScenarioBenchmarkResult:
    """Aggregated evaluation metrics across seeded episodes for one (agent, scenario)."""

    agent_name: str
    scenario: str
    num_episodes: int
    mean_reward: float
    std_reward: float
    mean_queue: float
    mean_wait_time_s: float
    std_wait_time_s: float
    mean_throughput: float
    std_throughput: float
    mean_switches: float
    inference_ms_per_step: float
    mean_collisions: float = 0.0


@dataclass(frozen=True)
class PharmacologyResult:
    """Quantitative metrics from real-time E/I gain shifts and neural lesioning."""

    condition: str
    g_exc: float
    g_inh: float
    lesion_target: str
    lesion_fraction: float
    active_neurons: int
    mean_reward: float
    mean_wait_time_s: float
    mean_throughput: float
    mean_switches: float
    mean_collisions: float = 0.0


class AgentEvaluator:
    """Evaluates traffic controllers across scenarios, latency, and neural perturbations."""

    def __init__(
        self,
        benchmark_config: BenchmarkConfig | None = None,
        env_config: TrafficEnvConfig | None = None,
    ) -> None:
        self.benchmark_config = benchmark_config or BenchmarkConfig()
        self.env_config = env_config or TrafficEnvConfig()

    def evaluate_agent_on_scenario(
        self,
        agent: BaseTrafficAgent,
        scenario: str,
        seeds: tuple[int, ...] | None = None,
    ) -> tuple[ScenarioBenchmarkResult, list[EpisodeMetrics]]:
        """Run deterministic multi-seed evaluation for one agent on one scenario."""
        eval_seeds = seeds or self.benchmark_config.eval_seeds
        episodes: list[EpisodeMetrics] = []

        for seed in eval_seeds:
            metrics = self._run_single_episode(
                agent=agent,
                scenario=scenario,
                seed=seed,
            )
            episodes.append(metrics)

        latency_ms = self.measure_inference_latency_ms(agent)

        rewards = [episode_metrics.total_reward for episode_metrics in episodes]
        queues = [episode_metrics.mean_queue for episode_metrics in episodes]
        wait_times = [episode_metrics.mean_wait_time_s for episode_metrics in episodes]
        throughputs = [episode_metrics.total_throughput for episode_metrics in episodes]
        switches = [episode_metrics.total_switches for episode_metrics in episodes]
        collisions = [episode_metrics.total_collisions for episode_metrics in episodes]

        summary = ScenarioBenchmarkResult(
            agent_name=agent.name,
            scenario=scenario,
            num_episodes=len(episodes),
            mean_reward=float(np.mean(rewards)),
            std_reward=float(np.std(rewards)),
            mean_queue=float(np.mean(queues)),
            mean_wait_time_s=float(np.mean(wait_times)),
            std_wait_time_s=float(np.std(wait_times)),
            mean_throughput=float(np.mean(throughputs)),
            std_throughput=float(np.std(throughputs)),
            mean_switches=float(np.mean(switches)),
            inference_ms_per_step=latency_ms,
            mean_collisions=float(np.mean(collisions)),
        )
        return summary, episodes

    def evaluate_all_agents(
        self,
        agents: list[BaseTrafficAgent],
    ) -> list[ScenarioBenchmarkResult]:
        """Evaluate every agent across all configured traffic scenarios."""
        results: list[ScenarioBenchmarkResult] = []
        for scenario in self.benchmark_config.scenarios:
            for agent in agents:
                summary, _ = self.evaluate_agent_on_scenario(agent, scenario)
                results.append(summary)
        return results

    def evaluate_pharmacology_and_lesions(
        self,
        fly_policy: FlyConnectomePolicy,
        scenario: str = "rush_hour",
    ) -> list[PharmacologyResult]:
        """Measure Fly Connectome PPO robustness under E/I modulation and regional lesions."""
        conditions: list[tuple[str, float, float, str, float]] = [
            ("Baseline (g_exc=1.0, g_inh=1.0)", 1.0, 1.0, "none", 0.0),
            ("High Excitation (g_exc=1.8, g_inh=1.0)", 1.8, 1.0, "none", 0.0),
            ("Low Inhibition (g_exc=1.0, g_inh=0.2)", 1.0, 0.2, "none", 0.0),
            ("High Inhibition (g_exc=1.0, g_inh=2.0)", 1.0, 2.0, "none", 0.0),
            ("CX Lesion 25%", 1.0, 1.0, "CX", 0.25),
            ("MB Lesion 25%", 1.0, 1.0, "MB", 0.25),
            ("Random Lesion 25%", 1.0, 1.0, "random", 0.25),
        ]

        results: list[PharmacologyResult] = []
        adapter = NeuralAgentAdapter(policy=fly_policy, agent_name="Fly Connectome PPO")
        circuit_groups = fly_policy.artifact.circuit_groups

        try:
            for label, g_exc, g_inh, target, frac in conditions:
                fly_policy.cell.set_pharmacology(g_exc=g_exc, g_inh=g_inh)
                fly_policy.cell.reset_lesion()

                if target in ("CX", "MB"):
                    fly_policy.cell.apply_region_lesion(
                        circuit_groups=circuit_groups,
                        target_group=target,
                        fraction=frac,
                        seed=42,
                    )
                elif target == "random":
                    fly_policy.cell.apply_random_lesion(fraction=frac, seed=42)

                summary, _ = self.evaluate_agent_on_scenario(adapter, scenario=scenario)
                results.append(
                    PharmacologyResult(
                        condition=label,
                        g_exc=g_exc,
                        g_inh=g_inh,
                        lesion_target=target,
                        lesion_fraction=frac,
                        active_neurons=fly_policy.cell.active_neuron_count(),
                        mean_reward=summary.mean_reward,
                        mean_wait_time_s=summary.mean_wait_time_s,
                        mean_throughput=summary.mean_throughput,
                        mean_switches=summary.mean_switches,
                        mean_collisions=summary.mean_collisions,
                    )
                )
        finally:
            fly_policy.cell.set_pharmacology(g_exc=1.0, g_inh=1.0)
            fly_policy.cell.reset_lesion()

        return results

    def measure_inference_latency_ms(self, agent: BaseTrafficAgent) -> float:
        """Benchmark single-step CPU action latency in milliseconds."""
        env = TrafficIntersectionEnv(config=self.env_config, profile_name="normal")
        obs, _ = env.reset(seed=999)
        agent.reset()

        for _ in range(self.benchmark_config.latency_warmup_steps):
            _ = agent.select_action(obs, deterministic=True)

        measurement_steps = self.benchmark_config.latency_measure_steps
        start_timestamp = time.perf_counter()
        for _ in range(measurement_steps):
            _ = agent.select_action(obs, deterministic=True)
        elapsed_seconds = time.perf_counter() - start_timestamp
        return float((elapsed_seconds / max(measurement_steps, 1)) * 1000.0)

    def _run_single_episode(
        self,
        agent: BaseTrafficAgent,
        scenario: str,
        seed: int,
    ) -> EpisodeMetrics:
        """Execute one deterministic episode and collect queue/wait/throughput statistics."""
        env = TrafficIntersectionEnv(config=self.env_config, profile_name=scenario)
        obs, _ = env.reset(seed=seed)
        agent.reset()

        total_reward = 0.0
        queue_trace: list[int] = []
        discharged_wait_times: list[float] = []
        max_wait_seen = 0.0
        done = False
        final_info: dict = {}

        while not done:
            action = agent.select_action(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            final_info = info

            total_reward += float(reward)
            queue_trace.append(int(info["total_queue"]))
            max_wait_seen = max(max_wait_seen, float(info["max_wait_time_s"]))
            discharged_wait_times.extend(float(wait_s) for wait_s in info["step_wait_times"])

        all_wait_times = discharged_wait_times + env.get_remaining_wait_times()
        mean_wait = float(np.mean(all_wait_times)) if all_wait_times else 0.0
        return EpisodeMetrics(
            scenario=scenario,
            seed=seed,
            agent_name=agent.name,
            total_reward=total_reward,
            mean_queue=float(np.mean(queue_trace)) if queue_trace else 0.0,
            mean_wait_time_s=mean_wait,
            max_wait_time_s=max_wait_seen,
            total_throughput=int(final_info.get("total_throughput", 0)),
            total_switches=int(final_info.get("total_switches", 0)),
            total_collisions=int(final_info.get("total_collisions", 0)),
        )

    @staticmethod
    def save_reports(
        output_dir: Path,
        scenario_results: list[ScenarioBenchmarkResult],
        pharmacology_results: list[PharmacologyResult],
    ) -> tuple[Path, Path]:
        """Write JSON and Markdown benchmark summaries using strictly factual headers."""
        output_dir.mkdir(parents=True, exist_ok=True)
        json_path = output_dir / "benchmark_summary.json"
        md_path = output_dir / "benchmark_table.md"

        payload = {
            "scenario_results": [asdict(item) for item in scenario_results],
            "pharmacology_results": [asdict(item) for item in pharmacology_results],
        }
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        lines: list[str] = [
            "# Benchmark Evaluation Results",
            "",
            "## Scenario Comparison",
            "",
            "| Scenario | Agent | Mean Reward | Mean Queue | Mean Wait (s) | Mean Throughput | Mean Switches | Mean Collisions | Inference (ms/step) |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for row in scenario_results:
            lines.append(
                f"| {row.scenario} | {row.agent_name} | "
                f"{row.mean_reward:.3f} +/- {row.std_reward:.3f} | "
                f"{row.mean_queue:.2f} | "
                f"{row.mean_wait_time_s:.2f} +/- {row.std_wait_time_s:.2f} | "
                f"{row.mean_throughput:.1f} +/- {row.std_throughput:.1f} | "
                f"{row.mean_switches:.1f} | "
                f"{row.mean_collisions:.2f} | "
                f"{row.inference_ms_per_step:.3f} |"
            )

        if pharmacology_results:
            lines.extend(
                [
                    "",
                    "## Pharmacology and Lesioning Results (rush_hour)",
                    "",
                    "| Condition | g_exc | g_inh | Active Neurons | Mean Reward | Mean Wait (s) | Mean Throughput | Mean Switches | Mean Collisions |",
                    "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
                ]
            )
            for row in pharmacology_results:
                lines.append(
                    f"| {row.condition} | {row.g_exc:.2f} | {row.g_inh:.2f} | "
                    f"{row.active_neurons} | {row.mean_reward:.3f} | "
                    f"{row.mean_wait_time_s:.2f} | {row.mean_throughput:.1f} | "
                    f"{row.mean_switches:.1f} | "
                    f"{row.mean_collisions:.2f} |"
                )

        md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return json_path, md_path
