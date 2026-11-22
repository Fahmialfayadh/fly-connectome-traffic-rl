"""Asyncio aiohttp web server uniting live simulation, training studio, and benchmarks."""

import asyncio
from contextlib import suppress
import json
from pathlib import Path
import re
from typing import Any

from aiohttp import WSMsgType, web

from fly_traffic.agents.base import BaseTrafficAgent, NeuralAgentAdapter
from fly_traffic.agents.baselines import load_policy_checkpoint
from fly_traffic.agents.fixed_timer import FixedTimerAgent
from fly_traffic.agents.fly_policy import FlyConnectomePolicy
from fly_traffic.config import (
    BenchmarkConfig,
    ConnectomeConfig,
    DEFAULT_ARTIFACTS_DIR,
    PPOConfig,
    PROJECT_ROOT,
    TrafficEnvConfig,
)
from fly_traffic.connectome.artifact import ConnectomeArtifact
from fly_traffic.connectome.shuffled import load_or_create_shuffled_connectome
from fly_traffic.evaluation.evaluator import AgentEvaluator
from fly_traffic.evaluation.visualizer import BenchmarkPlotter
from fly_traffic.web.simulation_service import SimulationService
from fly_traffic.web.training_service import TrainingJobManager


class TrafficLabServer:
    """HTTP + WebSocket server hosting the Light Instrument Lab frontend."""

    SPA_ROUTES: tuple[str, ...] = (
        "/",
        "/simulation",
        "/training",
        "/benchmark",
        "/architecture",
        "/docs",
        "/documentation",
    )
    _INCLUDE_PATTERN: re.Pattern[str] = re.compile(
        r"<!--\s*@include\s+([a-zA-Z0-9_./-]+)\s*-->"
    )

    def __init__(
        self,
        frontend_dir: Path | None = None,
        connectome_path: Path | None = None,
        checkpoints_dir: Path | None = None,
        benchmarks_dir: Path | None = None,
    ) -> None:
        self.frontend_dir = frontend_dir or (PROJECT_ROOT / "frontend")
        self.connectome_path = connectome_path or ConnectomeConfig.output_path
        self.checkpoints_dir = checkpoints_dir or PPOConfig.checkpoints_dir
        self.benchmarks_dir = benchmarks_dir or (DEFAULT_ARTIFACTS_DIR / "benchmarks")

        self.simulation_service = SimulationService(
            env_config=TrafficEnvConfig(),
            connectome_path=self.connectome_path,
            checkpoints_dir=self.checkpoints_dir,
        )
        self.training_manager = TrainingJobManager(
            simulation_service=self.simulation_service,
            connectome_path=self.connectome_path,
            checkpoints_dir=self.checkpoints_dir,
        )

        self._sim_clients: set[web.WebSocketResponse] = set()
        self._train_clients: set[web.WebSocketResponse] = set()
        self._sim_loop_task: asyncio.Task[None] | None = None
        self._benchmark_running: bool = False

    def create_app(self) -> web.Application:
        """Build aiohttp application with static routes, REST APIs, and WebSockets."""
        app = web.Application()
        app.on_startup.append(self._on_startup)
        app.on_cleanup.append(self._on_cleanup)

        for route_path in self.SPA_ROUTES:
            app.router.add_get(route_path, self._handle_index)
        app.router.add_get("/api/overview", self._handle_overview)
        app.router.add_post("/api/simulation/control", self._handle_simulation_control)
        app.router.add_post("/api/training/start", self._handle_training_start)
        app.router.add_post("/api/training/stop", self._handle_training_stop)
        app.router.add_post("/api/benchmark/run", self._handle_benchmark_run)
        app.router.add_get("/ws/simulation", self._handle_ws_simulation)
        app.router.add_get("/ws/training", self._handle_ws_training)

        css_dir = self.frontend_dir / "css"
        js_dir = self.frontend_dir / "js"
        assets_dir = self.frontend_dir / "assets"
        vendor_dir = self.frontend_dir / "vendor"
        if css_dir.exists():
            app.router.add_static("/css", css_dir)
        if js_dir.exists():
            app.router.add_static("/js", js_dir)
        if assets_dir.exists():
            app.router.add_static("/assets", assets_dir)
        if vendor_dir.exists():
            app.router.add_static("/vendor", vendor_dir)
        if self.benchmarks_dir.exists():
            app.router.add_static("/artifacts/benchmarks", self.benchmarks_dir)

        return app

    async def _on_startup(self, _app: web.Application) -> None:
        loop = asyncio.get_running_loop()
        self.training_manager.bind_event_loop(loop, self._schedule_training_broadcast)
        self._sim_loop_task = asyncio.create_task(self._simulation_broadcast_loop())

    async def _on_cleanup(self, _app: web.Application) -> None:
        if self.training_manager.is_training():
            self.training_manager.stop_job()

        if self._sim_loop_task is not None:
            self._sim_loop_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._sim_loop_task

        for ws in list(self._sim_clients) + list(self._train_clients):
            with suppress(Exception):
                await ws.close()

    def _render_index_html(self) -> str:
        """Assemble frontend/index.html by expanding <!-- @include partials/... --> markers."""
        index_path = self.frontend_dir / "index.html"
        root_resolved = self.frontend_dir.resolve()
        template = index_path.read_text(encoding="utf-8")

        def _replace_include(match: re.Match[str]) -> str:
            rel_path = match.group(1).strip()
            candidate = (self.frontend_dir / rel_path).resolve()
            if root_resolved not in candidate.parents or not candidate.is_file():
                raise FileNotFoundError(
                    f"Invalid or missing HTML partial '{rel_path}'."
                )
            return candidate.read_text(encoding="utf-8")

        return self._INCLUDE_PATTERN.sub(_replace_include, template)

    async def _handle_index(self, _request: web.Request) -> web.Response:
        html = self._render_index_html()
        return web.Response(
            text=html,
            content_type="text/html",
            charset="utf-8",
            headers={"Cache-Control": "no-cache"},
        )

    async def _handle_overview(self, _request: web.Request) -> web.Response:
        benchmark_json_path = self.benchmarks_dir / "benchmark_summary.json"
        benchmark_summary: dict[str, Any] | None = None
        if benchmark_json_path.exists():
            benchmark_summary = json.loads(
                benchmark_json_path.read_text(encoding="utf-8")
            )

        payload = {
            "simulation": self.simulation_service.get_frame_payload(),
            "overview": self.simulation_service.get_overview_payload(),
            "training": self.training_manager.get_state(),
            "benchmark_summary": benchmark_summary,
            "benchmark_running": self._benchmark_running,
        }
        return web.json_response(payload)

    async def _handle_simulation_control(self, request: web.Request) -> web.Response:
        try:
            body = await request.json()
            if not isinstance(body, dict):
                raise ValueError("Simulation control payload must be a JSON object.")
            frame = self._apply_simulation_command(body)
        except (ValueError, TypeError) as exc:
            return web.json_response({"error": str(exc)}, status=400)

        await self._broadcast_to_set(self._sim_clients, frame)
        return web.json_response(frame)

    async def _handle_training_start(self, request: web.Request) -> web.Response:
        try:
            body = await request.json()
            if not isinstance(body, dict):
                raise ValueError("Training payload must be a JSON object.")
            state = self.training_manager.start_job(
                model_type=str(self._body_value(body, "model_type", "fly")),
                total_timesteps=self._body_value(body, "total_timesteps", 40960),
                learning_rate=self._body_value(body, "learning_rate", 8e-4),
                seed=self._body_value(body, "seed", 42),
                save_checkpoint=bool(self._body_value(body, "save_checkpoint", True)),
                save_mode=body.get("save_mode"),
                custom_filename=body.get("custom_filename"),
                gamma=self._body_value(body, "gamma", PPOConfig.gamma),
                gae_lambda=self._body_value(body, "gae_lambda", PPOConfig.gae_lambda),
                clip_coef=self._body_value(body, "clip_coef", PPOConfig.clip_coef),
                ent_coef=self._body_value(body, "ent_coef", PPOConfig.ent_coef),
                vf_coef=self._body_value(body, "vf_coef", PPOConfig.vf_coef),
                update_epochs=self._body_value(
                    body,
                    "update_epochs",
                    PPOConfig.update_epochs,
                ),
                rollout_steps=self._body_value(
                    body,
                    "rollout_steps",
                    PPOConfig.rollout_steps,
                ),
                sequence_chunk_len=self._body_value(
                    body,
                    "sequence_chunk_len",
                    PPOConfig.sequence_chunk_len,
                ),
                num_minibatches=self._body_value(
                    body,
                    "num_minibatches",
                    PPOConfig.num_minibatches,
                ),
                anneal_lr=self._body_value(body, "anneal_lr", PPOConfig.anneal_lr),
                save_on_cancel=self._body_value(body, "save_on_cancel", True),
                resume_checkpoint=body.get("resume_checkpoint"),
            )
            return web.json_response(state)
        except RuntimeError as exc:
            return web.json_response({"error": str(exc)}, status=409)
        except (ValueError, TypeError) as exc:
            return web.json_response({"error": str(exc)}, status=400)

    async def _handle_training_stop(self, _request: web.Request) -> web.Response:
        state = self.training_manager.stop_job()
        return web.json_response(state)

    async def _handle_benchmark_run(self, request: web.Request) -> web.Response:
        if self._benchmark_running:
            return web.json_response(
                {"error": "Benchmark evaluation is already running."},
                status=409,
            )

        selected_agent_keys: list[str] | None = None
        if request.can_read_body:
            raw_body = await request.text()
            if raw_body.strip():
                try:
                    payload = json.loads(raw_body)
                except json.JSONDecodeError as exc:
                    return web.json_response(
                        {"error": f"Invalid JSON payload: {exc}"},
                        status=400,
                    )
                if not isinstance(payload, dict):
                    return web.json_response(
                        {"error": "Benchmark request payload must be a JSON object."},
                        status=400,
                    )
                raw_keys = payload.get("agent_keys")
                if raw_keys is not None:
                    if not isinstance(raw_keys, list) or not all(
                        isinstance(k, str) and k.strip() for k in raw_keys
                    ):
                        return web.json_response(
                            {"error": "agent_keys must be a list of model keys."},
                            status=400,
                        )
                    selected_agent_keys = list(dict.fromkeys(k.strip() for k in raw_keys))
                    if not selected_agent_keys:
                        return web.json_response(
                            {"error": "Pilih minimal satu model untuk menjalankan benchmark."},
                            status=400,
                        )

        self._benchmark_running = True
        try:
            summary_payload = await asyncio.to_thread(
                self._run_benchmark_sync,
                selected_agent_keys,
            )
            return web.json_response(summary_payload)
        except ValueError as exc:
            return web.json_response({"error": str(exc)}, status=400)
        finally:
            self._benchmark_running = False

    async def _handle_ws_simulation(self, request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse(heartbeat=20.0)
        await ws.prepare(request)
        self._sim_clients.add(ws)

        try:
            await ws.send_json(self.simulation_service.get_frame_payload())
            async for message in ws:
                if message.type == WSMsgType.TEXT:
                    try:
                        payload = json.loads(message.data)
                        if not isinstance(payload, dict):
                            raise ValueError("Simulation control payload must be a JSON object.")
                        frame = self._apply_simulation_command(payload)
                        await self._broadcast_to_set(self._sim_clients, frame)
                    except (ValueError, TypeError, json.JSONDecodeError) as exc:
                        await ws.send_json({"type": "error", "error": str(exc)})
                elif message.type == WSMsgType.ERROR:
                    break
        finally:
            self._sim_clients.discard(ws)

        return ws

    async def _handle_ws_training(self, request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse(heartbeat=20.0)
        await ws.prepare(request)
        self._train_clients.add(ws)

        try:
            await ws.send_json(self.training_manager.get_state())
            async for message in ws:
                if message.type == WSMsgType.ERROR:
                    break
        finally:
            self._train_clients.discard(ws)

        return ws

    def _apply_simulation_command(self, body: dict[str, Any]) -> dict[str, Any]:
        action = str(body.get("action", "")).strip().lower()

        if action == "step":
            return self.simulation_service.step_once()
        if action == "reset":
            return self.simulation_service.reset_simulation(seed=body.get("seed", 42))
        if action == "set_running":
            return self.simulation_service.set_running(bool(body.get("running", True)))
        if action == "set_speed":
            return self.simulation_service.set_speed(
                float(body.get("steps_per_second", 2.0))
            )
        if action == "set_scenario":
            return self.simulation_service.set_scenario(str(body.get("scenario", "rush_hour")))
        if action == "set_safety_buffer":
            all_red_steps = self._parse_required_int(
                body.get("all_red_steps", 1),
                "all_red_steps",
            )
            return self.simulation_service.set_safety_buffer(all_red_steps)
        if action == "set_continuous_mode":
            return self.simulation_service.set_continuous_mode(
                bool(body.get("enabled", True))
            )
        if action == "spawn_vehicles":
            lane_id = self._parse_required_int(body.get("lane_id"), "lane_id")
            raw_count = body.get("count", 1)
            count = self._parse_required_int(
                1 if raw_count is None else raw_count,
                "count",
            )
            return self.simulation_service.spawn_vehicles(lane_id=lane_id, count=count)
        if action == "spawn_random":
            raw_count = body.get("count", 5)
            count = self._parse_required_int(
                5 if raw_count is None else raw_count,
                "count",
            )
            return self.simulation_service.spawn_random_vehicles(count=count)
        if action == "spawn_surge":
            raw_count = (
                body.get("count_per_lane")
                if body.get("count_per_lane") is not None
                else body.get("count", 3)
            )
            count_per_lane = self._parse_required_int(
                3 if raw_count is None else raw_count,
                "count_per_lane",
            )
            return self.simulation_service.spawn_surge(count_per_lane=count_per_lane)
        if action == "set_agent":
            return self.simulation_service.set_agent(str(body.get("agent_key", "fly")))
        if action == "set_pharmacology":
            return self.simulation_service.set_pharmacology(
                g_exc=float(body.get("g_exc", 1.0)),
                g_inh=float(body.get("g_inh", 1.0)),
            )
        if action == "apply_lesion":
            return self.simulation_service.apply_lesion(
                target=str(body.get("target", "CX")),
                fraction=float(body.get("fraction", 0.25)),
            )
        if action == "reset_brain":
            return self.simulation_service.reset_brain()

        raise ValueError(f"Unsupported simulation action '{action}'.")

    @staticmethod
    def _body_value(body: dict[str, Any], key: str, default: Any) -> Any:
        value = body.get(key, default)
        return default if value is None else value

    @staticmethod
    def _parse_required_int(value: Any, name: str) -> int:
        if (
            value is None
            or isinstance(value, bool)
            or (isinstance(value, float) and not value.is_integer())
        ):
            raise ValueError(f"{name} must be an integer.")
        return int(value)

    async def _simulation_broadcast_loop(self) -> None:
        while True:
            sps = self.simulation_service.steps_per_second
            interval = 1.0 / max(sps, 0.5)
            await asyncio.sleep(interval)

            if self.simulation_service.is_running and self._sim_clients:
                frame = self.simulation_service.step_once()
                await self._broadcast_to_set(self._sim_clients, frame)

    def _schedule_training_broadcast(self, payload: dict[str, Any]) -> None:
        asyncio.create_task(self._broadcast_training_event(payload))

    async def _broadcast_training_event(self, payload: dict[str, Any]) -> None:
        await self._broadcast_to_set(self._train_clients, payload)
        event_type = payload.get("type")
        should_refresh_sim = event_type == "training_completed" or (
            event_type == "training_cancelled"
            and bool(payload.get("saved_checkpoint_file"))
        )
        if should_refresh_sim:
            overview = self.simulation_service.get_overview_payload()
            await self._broadcast_to_set(
                self._sim_clients,
                {"type": "overview_updated", "overview": overview},
            )
            frame = self.simulation_service.get_frame_payload()
            await self._broadcast_to_set(self._sim_clients, frame)

    @staticmethod
    async def _broadcast_to_set(
        clients: set[web.WebSocketResponse],
        payload: dict[str, Any],
    ) -> None:
        if not clients:
            return
        text_data = json.dumps(payload)
        dead_clients: list[web.WebSocketResponse] = []
        for ws in list(clients):
            if ws.closed:
                dead_clients.append(ws)
                continue
            try:
                await ws.send_str(text_data)
            except Exception:
                dead_clients.append(ws)
        for dead in dead_clients:
            clients.discard(dead)

    def _run_benchmark_sync(
        self,
        selected_agent_keys: list[str] | None = None,
    ) -> dict[str, Any]:
        artifact = (
            ConnectomeArtifact.load(self.connectome_path)
            if self.connectome_path.exists()
            else None
        )
        shuffled_artifact = (
            load_or_create_shuffled_connectome(self.connectome_path)
            if self.connectome_path.exists()
            else None
        )
        agents: list[BaseTrafficAgent] = []
        histories: dict[str, dict[str, list[float]]] = {}
        fly_policy: FlyConnectomePolicy | None = None

        if selected_agent_keys is None:
            agents.append(FixedTimerAgent())
            for key, label in [
                ("mlp", "PPO-MLP"),
                ("gru", "PPO-GRU"),
                ("fly_shuffled", "Shuffled Connectome PPO"),
                ("fly", "Fly Connectome PPO"),
            ]:
                ckpt = self.checkpoints_dir / f"{key}_ppo.pt"
                if not ckpt.exists():
                    continue
                policy, payload = load_policy_checkpoint(
                    path=ckpt,
                    artifact=artifact if key == "fly" else None,
                    shuffled_artifact=shuffled_artifact if key == "fly_shuffled" else None,
                )
                agents.append(NeuralAgentAdapter(policy=policy, agent_name=label))
                if isinstance(payload.get("history"), dict):
                    histories[label] = payload["history"]
                if key == "fly" and isinstance(policy, FlyConnectomePolicy):
                    fly_policy = policy
        else:
            inventory = self.simulation_service.get_overview_payload().get("checkpoints", {})
            used_names: set[str] = set()
            for key in selected_agent_keys:
                if key == "fixed_timer":
                    agents.append(FixedTimerAgent())
                    used_names.add("Fixed Timer")
                    continue

                info = inventory.get(key)
                if not info:
                    raise ValueError(f"Model '{key}' tidak ditemukan di inventori.")

                model_type = str(info.get("model_type", ""))
                if not info.get("exists"):
                    if key in ("fly", "fly_shuffled"):
                        target_artifact = shuffled_artifact if key == "fly_shuffled" else artifact
                        if target_artifact is None:
                            raise ValueError(f"Connectome artifact tidak tersedia untuk '{key}'.")
                        policy = FlyConnectomePolicy(
                            obs_dim=22,
                            act_dim=4,
                            artifact=target_artifact,
                        )
                        policy.eval()
                        label = str(info.get("label") or f"{key} (untrained)")
                        if label in used_names:
                            label = f"{label} ({key})"
                        used_names.add(label)
                        agents.append(NeuralAgentAdapter(policy=policy, agent_name=label))
                        if fly_policy is None and model_type == "fly":
                            fly_policy = policy
                        continue
                    raise ValueError(f"Model '{key}' belum memiliki checkpoint terlatih.")

                ckpt_file = str(info.get("checkpoint_file", ""))
                ckpt_path = self.checkpoints_dir / ckpt_file
                if not ckpt_path.exists():
                    raise ValueError(f"Checkpoint untuk model '{key}' tidak ditemukan: {ckpt_file}")

                policy, payload = load_policy_checkpoint(
                    path=ckpt_path,
                    artifact=artifact if model_type == "fly" else None,
                    shuffled_artifact=shuffled_artifact if model_type == "fly_shuffled" else None,
                )
                label = str(info.get("label") or key)
                if label in used_names:
                    label = f"{label} ({ckpt_file})"
                used_names.add(label)

                agents.append(NeuralAgentAdapter(policy=policy, agent_name=label))
                if isinstance(payload.get("history"), dict):
                    histories[label] = payload["history"]
                if fly_policy is None and model_type == "fly" and isinstance(policy, FlyConnectomePolicy):
                    fly_policy = policy

        if not agents:
            raise ValueError("Tidak ada model valid yang dipilih untuk benchmark.")

        evaluator = AgentEvaluator(
            benchmark_config=BenchmarkConfig(output_dir=self.benchmarks_dir),
            env_config=TrafficEnvConfig(),
        )
        scenario_results = evaluator.evaluate_all_agents(agents)
        pharmacology_results = (
            evaluator.evaluate_pharmacology_and_lesions(fly_policy)
            if fly_policy is not None
            else []
        )
        json_path, _ = evaluator.save_reports(
            output_dir=self.benchmarks_dir,
            scenario_results=scenario_results,
            pharmacology_results=pharmacology_results,
        )

        if histories:
            BenchmarkPlotter.plot_training_curves(
                histories=histories,
                output_path=self.benchmarks_dir / "training_curves.png",
            )
        BenchmarkPlotter.plot_scenario_comparison(
            results=scenario_results,
            output_path=self.benchmarks_dir / "scenario_comparison.png",
        )
        if pharmacology_results:
            BenchmarkPlotter.plot_pharmacology_ablation(
                results=pharmacology_results,
                output_path=self.benchmarks_dir / "pharmacology_ablation.png",
            )

        return json.loads(json_path.read_text(encoding="utf-8"))
