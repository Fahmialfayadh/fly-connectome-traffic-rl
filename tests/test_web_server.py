"""Integration tests for the aiohttp Web Lab APIs, WebSockets, and training job manager."""

import asyncio
from pathlib import Path
import threading

from aiohttp.test_utils import TestClient, TestServer
import torch

from fly_traffic.agents.baselines import (
    build_policy,
    load_policy_checkpoint,
    save_policy_checkpoint,
)
from fly_traffic.config import PPOConfig, PROJECT_ROOT, TrafficEnvConfig
from fly_traffic.training.ppo_trainer import PPOTrainer
from fly_traffic.web.server import TrafficLabServer
from fly_traffic.web.simulation_service import SimulationService
from tests.test_policies import _make_mock_artifact


def test_web_server_rest_and_websockets(tmp_path: Path) -> None:
    """Verify overview, simulation WebSocket/REST controls, and non-destructive training."""

    async def _run_checks() -> None:
        conn_path = tmp_path / "connectome.pt"
        ckpt_dir = tmp_path / "checkpoints"
        bench_dir = tmp_path / "benchmarks"
        artifact = _make_mock_artifact(num_neurons=100)
        artifact.save(conn_path)

        server = TrafficLabServer(
            frontend_dir=PROJECT_ROOT / "frontend",
            connectome_path=conn_path,
            checkpoints_dir=ckpt_dir,
            benchmarks_dir=bench_dir,
        )
        app = server.create_app()

        async with TestClient(TestServer(app)) as client:
            # 1. Static HTML shell & 3D brain assets
            resp_index = await client.get("/")
            assert resp_index.status == 200
            html = await resp_index.text()
            assert "Drosophila Connectome Traffic Lab" in html
            assert 'id="panel-docs"' in html
            assert "Dorkenwald" in html
            assert "Schlegel" in html

            resp_neurons = await client.get("/assets/brain/fly_brain_neurons.json")
            assert resp_neurons.status == 200
            neurons_meta = await resp_neurons.json()
            assert neurons_meta["num_neurons"] == 2500

            resp_glb = await client.get("/assets/brain/fly_brain_meshes.glb")
            assert resp_glb.status == 200

            resp_three = await client.get("/vendor/three/three.module.min.js")
            assert resp_three.status == 200

            # 2. Overview payload
            resp_overview = await client.get("/api/overview")
            assert resp_overview.status == 200
            overview = await resp_overview.json()
            assert overview["simulation"]["mode"] == "INFERENCE"
            assert overview["overview"]["connectome_metadata"]["num_neurons"] == 100

            # 3. Simulation controls (step, pharmacology, lesion, reset, switch agent)
            resp_step = await client.post("/api/simulation/control", json={"action": "step"})
            assert resp_step.status == 200
            step_data = await resp_step.json()
            assert step_data["snapshot"]["step"] == 1
            assert len(step_data["policy_telemetry"]["action_probabilities"]) == 4

            resp_pharma = await client.post(
                "/api/simulation/control",
                json={"action": "set_pharmacology", "g_exc": 1.6, "g_inh": 0.5},
            )
            assert resp_pharma.status == 200
            pharma_data = await resp_pharma.json()
            assert abs(pharma_data["connectome_telemetry"]["g_exc"] - 1.6) < 1e-5
            assert abs(pharma_data["connectome_telemetry"]["g_inh"] - 0.5) < 1e-5

            resp_lesion = await client.post(
                "/api/simulation/control",
                json={"action": "apply_lesion", "target": "CX", "fraction": 0.25},
            )
            assert resp_lesion.status == 200
            lesion_data = await resp_lesion.json()
            assert lesion_data["connectome_telemetry"]["active_neurons"] < 100

            resp_timer = await client.post(
                "/api/simulation/control",
                json={"action": "set_agent", "agent_key": "fixed_timer"},
            )
            assert resp_timer.status == 200
            timer_data = await resp_timer.json()
            assert timer_data["agent_key"] == "fixed_timer"
            assert timer_data["connectome_telemetry"]["is_fly_active"] is False
            assert timer_data["model_telemetry"]["kind"] == "fixed_timer"
            assert timer_data["model_telemetry"]["phase_durations"] == [8, 4, 8, 4]

            resp_buffer = await client.post(
                "/api/simulation/control",
                json={"action": "set_safety_buffer", "all_red_steps": 0},
            )
            assert resp_buffer.status == 200
            buffer_data = await resp_buffer.json()
            assert buffer_data["snapshot"]["all_red_steps"] == 0
            assert "total_collisions" in buffer_data["snapshot"]

            assert buffer_data["snapshot"]["continuous_mode"] is True
            resp_finite = await client.post(
                "/api/simulation/control",
                json={"action": "set_continuous_mode", "enabled": False},
            )
            assert resp_finite.status == 200
            assert (await resp_finite.json())["snapshot"]["continuous_mode"] is False

            resp_cont = await client.post(
                "/api/simulation/control",
                json={"action": "set_continuous_mode", "enabled": True},
            )
            assert resp_cont.status == 200
            assert (await resp_cont.json())["snapshot"]["continuous_mode"] is True

            # 4. Simulation WebSocket receives frame
            async with client.ws_connect("/ws/simulation") as sim_ws:
                msg = await sim_ws.receive_json()
                assert msg["type"] == "simulation_frame"

            # 5. Live Training Studio job + WebSocket update streaming
            async with client.ws_connect("/ws/training") as train_ws:
                init_state = await train_ws.receive_json()
                assert init_state["type"] == "training_state"

                start_resp = await client.post(
                    "/api/training/start",
                    json={
                        "model_type": "mlp",
                        "total_timesteps": 2048,
                        "learning_rate": 0.0008,
                        "seed": 42,
                        "save_checkpoint": False,
                    },
                )
                assert start_resp.status == 200

                saw_update = False
                saw_completed = False
                for _ in range(30):
                    event = await asyncio.wait_for(train_ws.receive_json(), timeout=15.0)
                    if event.get("type") == "training_update":
                        saw_update = True
                        preview_frames = event.get("preview_frames")
                        assert isinstance(preview_frames, list)
                        assert len(preview_frames) == 4
                        assert all(
                            "lane_queues" in frame and "total_collisions" in frame
                            for frame in preview_frames
                        )
                    if event.get("type") == "training_completed":
                        saw_completed = True
                        break

                assert saw_update
                assert saw_completed

    asyncio.run(_run_checks())


def test_custom_checkpoint_creation_and_provenance(tmp_path: Path) -> None:
    """Verify save_mode='new_file', WebSocket broadcast, overview, and set_agent."""

    async def _run_checks() -> None:
        conn_path = tmp_path / "connectome.pt"
        ckpt_dir = tmp_path / "checkpoints"
        bench_dir = tmp_path / "benchmarks"
        artifact = _make_mock_artifact(num_neurons=100)
        artifact.save(conn_path)

        server = TrafficLabServer(
            frontend_dir=PROJECT_ROOT / "frontend",
            connectome_path=conn_path,
            checkpoints_dir=ckpt_dir,
            benchmarks_dir=bench_dir,
        )
        app = server.create_app()

        async with TestClient(TestServer(app)) as client:
            # Pause continuous simulation loop so WS messages reflect training completion cleanly.
            pause_resp = await client.post(
                "/api/simulation/control",
                json={"action": "set_running", "running": False},
            )
            assert pause_resp.status == 200

            # Invalid save_mode must return HTTP 400.
            bad_mode_resp = await client.post(
                "/api/training/start",
                json={
                    "model_type": "mlp",
                    "total_timesteps": 1024,
                    "save_mode": "invalid_mode",
                },
            )
            assert bad_mode_resp.status == 400

            async with (
                client.ws_connect("/ws/simulation") as sim_ws,
                client.ws_connect("/ws/training") as train_ws,
            ):
                initial_sim = await sim_ws.receive_json()
                assert initial_sim["type"] == "simulation_frame"

                initial_train = await train_ws.receive_json()
                assert initial_train["type"] == "training_state"

                start_resp = await client.post(
                    "/api/training/start",
                    json={
                        "model_type": "mlp",
                        "total_timesteps": 2048,
                        "learning_rate": 0.0008,
                        "seed": 42,
                        "save_mode": "new_file",
                        "custom_filename": "mlp_experiment_01",
                    },
                )
                assert start_resp.status == 200
                start_payload = await start_resp.json()
                assert start_payload["config"]["save_mode"] == "new_file"
                assert (
                    start_payload["config"]["target_checkpoint_file"]
                    == "mlp_experiment_01.pt"
                )

                completed_event: dict | None = None
                for _ in range(30):
                    event = await asyncio.wait_for(
                        train_ws.receive_json(),
                        timeout=15.0,
                    )
                    if event.get("type") == "training_completed":
                        completed_event = event
                        break

                assert completed_event is not None
                assert completed_event["saved_checkpoint_file"] == "mlp_experiment_01.pt"
                assert completed_event["saved_agent_key"] == "ckpt:mlp_experiment_01"

                assert (ckpt_dir / "mlp_experiment_01.pt").exists()

                saw_overview_updated = False
                saw_activated_frame = False
                for _ in range(10):
                    sim_event = await asyncio.wait_for(
                        sim_ws.receive_json(),
                        timeout=5.0,
                    )
                    if sim_event.get("type") == "overview_updated":
                        saw_overview_updated = True
                        assert (
                            "ckpt:mlp_experiment_01"
                            in sim_event["overview"]["checkpoints"]
                        )
                    elif (
                        sim_event.get("type") == "simulation_frame"
                        and sim_event.get("agent_key") == "ckpt:mlp_experiment_01"
                    ):
                        saw_activated_frame = True
                    if saw_overview_updated and saw_activated_frame:
                        break

                assert saw_overview_updated
                assert saw_activated_frame

            resp_overview = await client.get("/api/overview")
            assert resp_overview.status == 200
            overview_data = await resp_overview.json()
            checkpoints = overview_data["overview"]["checkpoints"]
            assert "ckpt:mlp_experiment_01" in checkpoints
            custom_entry = checkpoints["ckpt:mlp_experiment_01"]
            assert custom_entry["checkpoint_file"] == "mlp_experiment_01.pt"
            assert custom_entry["is_custom"] is True
            assert custom_entry["model_type"] == "mlp"
            assert (
                custom_entry["relative_path"]
                == "artifacts/checkpoints/mlp_experiment_01.pt"
            )
            assert custom_entry["exists"] is True

            assert checkpoints["fly"]["is_custom"] is False
            assert checkpoints["fly"]["model_type"] == "fly"
            assert checkpoints["fly"]["relative_path"] == "Not saved yet"
            assert checkpoints["fixed_timer"]["model_type"] == "fixed_timer"
            assert checkpoints["fixed_timer"]["relative_path"] == "rule-based"
            assert checkpoints["fixed_timer"]["is_custom"] is False
            assert list(checkpoints.keys())[-1] == "fixed_timer"

            # Switch away and back to the custom checkpoint via REST control.
            resp_timer = await client.post(
                "/api/simulation/control",
                json={"action": "set_agent", "agent_key": "fixed_timer"},
            )
            assert resp_timer.status == 200

            resp_set_custom = await client.post(
                "/api/simulation/control",
                json={"action": "set_agent", "agent_key": "ckpt:mlp_experiment_01"},
            )
            assert resp_set_custom.status == 200
            custom_frame = await resp_set_custom.json()
            assert custom_frame["agent_key"] == "ckpt:mlp_experiment_01"
            assert (
                custom_frame["checkpoint_provenance"]["checkpoint_file"]
                == "mlp_experiment_01.pt"
            )

            # Collision avoidance and filename sanitization when mlp_experiment_01.pt exists.
            collision_resp = await client.post(
                "/api/training/start",
                json={
                    "model_type": "mlp",
                    "total_timesteps": 2048,
                    "save_mode": "new_file",
                    "custom_filename": "../mlp_experiment_01.pt",
                },
            )
            assert collision_resp.status == 200
            collision_payload = await collision_resp.json()
            assert (
                collision_payload["config"]["target_checkpoint_file"]
                == "mlp_experiment_01_2.pt"
            )

            for _ in range(30):
                state = server.training_manager.get_state()
                if state["status"] in ("completed", "error", "cancelled"):
                    break
                await asyncio.sleep(0.1)
            assert server.training_manager.get_state()["status"] == "completed"
            assert (ckpt_dir / "mlp_experiment_01_2.pt").exists()

    asyncio.run(_run_checks())


def test_checkpoint_directory_scanning_and_custom_fly_controls(tmp_path: Path) -> None:
    """Verify startup .pt scanning, non-policy skipping, order, and custom Fly controls."""
    conn_path = tmp_path / "connectome.pt"
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    artifact = _make_mock_artifact(num_neurons=100)
    artifact.save(conn_path)

    fly_policy = build_policy("fly", obs_dim=22, act_dim=4, artifact=artifact)
    save_policy_checkpoint(
        path=ckpt_dir / "fly_variant_b.pt",
        model_type="fly",
        policy=fly_policy,
        obs_dim=22,
        act_dim=4,
        connectome_path=conn_path,
        trained_timesteps=4096,
    )
    mlp_policy = build_policy("mlp", obs_dim=22, act_dim=4)
    save_policy_checkpoint(
        path=ckpt_dir / "alpha_mlp.pt",
        model_type="mlp",
        policy=mlp_policy,
        obs_dim=22,
        act_dim=4,
        trained_timesteps=2048,
    )
    # Non-policy .pt artifact in checkpoints_dir should be ignored cleanly.
    torch.save({"not_a_policy": [1, 2, 3]}, ckpt_dir / "unrelated_artifact.pt")

    service = SimulationService(
        connectome_path=conn_path,
        checkpoints_dir=ckpt_dir,
    )
    overview = service.get_overview_payload()
    keys = list(overview["checkpoints"].keys())
    assert keys == [
        "fly",
        "fly_shuffled",
        "gru",
        "mlp",
        "ckpt:alpha_mlp",
        "ckpt:fly_variant_b",
        "fixed_timer",
    ]

    fly_custom = overview["checkpoints"]["ckpt:fly_variant_b"]
    assert fly_custom["is_custom"] is True
    assert fly_custom["model_type"] == "fly"
    assert fly_custom["checkpoint_file"] == "fly_variant_b.pt"
    assert fly_custom["relative_path"] == "artifacts/checkpoints/fly_variant_b.pt"

    service.set_agent("ckpt:fly_variant_b")
    pharma_frame = service.set_pharmacology(g_exc=1.5, g_inh=0.7)
    assert pharma_frame["connectome_telemetry"]["is_fly_active"] is True
    assert abs(pharma_frame["connectome_telemetry"]["g_exc"] - 1.5) < 1e-5
    assert abs(pharma_frame["connectome_telemetry"]["g_inh"] - 0.7) < 1e-5

    lesion_frame = service.apply_lesion("CX", fraction=0.25)
    assert lesion_frame["connectome_telemetry"]["active_neurons"] < 100

    # Default 'fly' must remain untouched by perturbations applied to custom Fly model.
    default_fly_frame = service.set_agent("fly")
    assert abs(default_fly_frame["connectome_telemetry"]["g_exc"] - 1.0) < 1e-5
    assert default_fly_frame["connectome_telemetry"]["active_neurons"] == 100
    assert default_fly_frame["model_telemetry"]["kind"] == "fly"

    # Custom MLP must expose real layer-1 and layer-2 telemetry after stepping.
    service.set_agent("ckpt:alpha_mlp")
    mlp_frame = service.step_once()
    assert mlp_frame["connectome_telemetry"]["is_fly_active"] is False
    assert mlp_frame["model_telemetry"]["kind"] == "mlp"
    assert mlp_frame["model_telemetry"]["layer1_b64"]
    assert mlp_frame["model_telemetry"]["layer2_b64"]
    assert mlp_frame["model_telemetry"]["layer1_mean_abs"] > 0.0

    # Shuffled connectome null model must identify itself accurately.
    shuffled_frame = service.set_agent("fly_shuffled")
    assert shuffled_frame["connectome_telemetry"]["is_fly_active"] is True
    assert shuffled_frame["model_telemetry"]["kind"] == "fly_shuffled"
    assert shuffled_frame["model_telemetry"]["is_shuffled"] is True


def test_live_simulation_mode_maximal_and_manual_spawns(tmp_path: Path) -> None:
    """Verify Mode Maximal (80/lane = 640 total) and manual vehicle spawn controls."""

    async def _run_checks() -> None:
        conn_path = tmp_path / "connectome.pt"
        ckpt_dir = tmp_path / "checkpoints"
        bench_dir = tmp_path / "benchmarks"
        artifact = _make_mock_artifact(num_neurons=100)
        artifact.save(conn_path)

        server = TrafficLabServer(
            frontend_dir=PROJECT_ROOT / "frontend",
            connectome_path=conn_path,
            checkpoints_dir=ckpt_dir,
            benchmarks_dir=bench_dir,
        )
        app = server.create_app()

        async with TestClient(TestServer(app)) as client:
            pause_resp = await client.post(
                "/api/simulation/control",
                json={"action": "set_running", "running": False},
            )
            assert pause_resp.status == 200
            paused_frame = await pause_resp.json()
            assert paused_frame["snapshot"]["max_lane_capacity"] == 80
            assert paused_frame["snapshot"]["max_total_capacity"] == 640

            # Spawn 15 vehicles on lane 2.
            spawn_resp = await client.post(
                "/api/simulation/control",
                json={"action": "spawn_vehicles", "lane_id": 2, "count": 15},
            )
            assert spawn_resp.status == 200
            spawn_frame = await spawn_resp.json()
            assert spawn_frame["snapshot"]["lane_queues"][2] == 15
            assert spawn_frame["snapshot"]["manual_spawns_per_lane"][2] == 15

            # Clamp per-call count at 40 and saturate lane 2 hard cap at 80.
            resp_fill_1 = await client.post(
                "/api/simulation/control",
                json={"action": "spawn_vehicles", "lane_id": 2, "count": 99},
            )
            assert resp_fill_1.status == 200
            fill_1_frame = await resp_fill_1.json()
            assert fill_1_frame["snapshot"]["lane_queues"][2] == 55
            assert fill_1_frame["snapshot"]["manual_spawns_per_lane"][2] == 40

            resp_fill_2 = await client.post(
                "/api/simulation/control",
                json={"action": "spawn_vehicles", "lane_id": 2, "count": 40},
            )
            assert resp_fill_2.status == 200
            fill_2_frame = await resp_fill_2.json()
            assert fill_2_frame["snapshot"]["lane_queues"][2] == 80
            assert fill_2_frame["snapshot"]["manual_spawns_per_lane"][2] == 25

            # Invalid lane_id must return HTTP 400.
            bad_lane_resp = await client.post(
                "/api/simulation/control",
                json={"action": "spawn_vehicles", "lane_id": 8, "count": 5},
            )
            assert bad_lane_resp.status == 400

            # Reset clears queues while preserving 80/lane = 640 capacity.
            reset_resp = await client.post(
                "/api/simulation/control",
                json={"action": "reset", "seed": 42},
            )
            assert reset_resp.status == 200
            reset_frame = await reset_resp.json()
            assert reset_frame["snapshot"]["total_queue"] == 0
            assert reset_frame["snapshot"]["max_lane_capacity"] == 80
            assert reset_frame["snapshot"]["max_total_capacity"] == 640

            # Random spawn distributes exact count across lanes 0..7.
            random_resp = await client.post(
                "/api/simulation/control",
                json={"action": "spawn_random", "count": 12},
            )
            assert random_resp.status == 200
            random_frame = await random_resp.json()
            assert sum(random_frame["snapshot"]["manual_spawns_per_lane"]) == 12
            assert random_frame["snapshot"]["total_queue"] == 12

            # Surge spawn injects across all 8 lanes.
            surge_resp = await client.post(
                "/api/simulation/control",
                json={"action": "spawn_surge", "count_per_lane": 4},
            )
            assert surge_resp.status == 200
            surge_frame = await surge_resp.json()
            assert surge_frame["snapshot"]["manual_spawns_per_lane"] == [4] * 8
            assert surge_frame["snapshot"]["total_queue"] == 44

            # Advancing one simulation step clears manual_spawns_per_lane.
            step_resp = await client.post(
                "/api/simulation/control",
                json={"action": "step"},
            )
            assert step_resp.status == 200
            step_frame = await step_resp.json()
            assert step_frame["snapshot"]["manual_spawns_per_lane"] == [0] * 8

    asyncio.run(_run_checks())


def test_custom_ppo_hyperparams_validation_and_run(tmp_path: Path) -> None:
    """Verify HTTP 400 on invalid PPO hyperparameters and HTTP 200 on valid custom runs."""

    async def _run_checks() -> None:
        conn_path = tmp_path / "connectome.pt"
        ckpt_dir = tmp_path / "checkpoints"
        bench_dir = tmp_path / "benchmarks"
        artifact = _make_mock_artifact(num_neurons=100)
        artifact.save(conn_path)

        server = TrafficLabServer(
            frontend_dir=PROJECT_ROOT / "frontend",
            connectome_path=conn_path,
            checkpoints_dir=ckpt_dir,
            benchmarks_dir=bench_dir,
        )
        app = server.create_app()

        async with TestClient(TestServer(app)) as client:
            invalid_payloads = [
                {
                    "model_type": "mlp",
                    "total_timesteps": 1024,
                    "rollout_steps": 32,
                    "sequence_chunk_len": 24,
                    "num_minibatches": 4,
                },
                {
                    "model_type": "mlp",
                    "total_timesteps": 1024,
                    "rollout_steps": 64,
                    "sequence_chunk_len": 16,
                    "num_minibatches": 3,
                },
                {
                    "model_type": "mlp",
                    "total_timesteps": 1024,
                    "rollout_steps": 48,
                    "sequence_chunk_len": 16,
                    "num_minibatches": 4,
                },
                {
                    "model_type": "mlp",
                    "total_timesteps": 1024,
                    "gamma": 1.2,
                },
                {
                    "model_type": "mlp",
                    "total_timesteps": 512,
                },
            ]
            for invalid_payload in invalid_payloads:
                bad_resp = await client.post("/api/training/start", json=invalid_payload)
                assert bad_resp.status == 400

            valid_resp = await client.post(
                "/api/training/start",
                json={
                    "model_type": "mlp",
                    "total_timesteps": 1024,
                    "learning_rate": 0.0005,
                    "seed": 7,
                    "gamma": 0.95,
                    "gae_lambda": 0.90,
                    "clip_coef": 0.15,
                    "ent_coef": 0.01,
                    "vf_coef": 0.5,
                    "update_epochs": 2,
                    "rollout_steps": 64,
                    "sequence_chunk_len": 16,
                    "num_minibatches": 4,
                    "anneal_lr": False,
                    "save_on_cancel": True,
                    "save_mode": "none",
                },
            )
            assert valid_resp.status == 200
            start_data = await valid_resp.json()
            cfg = start_data["config"]
            assert abs(cfg["gamma"] - 0.95) < 1e-6
            assert abs(cfg["gae_lambda"] - 0.90) < 1e-6
            assert abs(cfg["clip_coef"] - 0.15) < 1e-6
            assert abs(cfg["ent_coef"] - 0.01) < 1e-6
            assert abs(cfg["vf_coef"] - 0.5) < 1e-6
            assert cfg["update_epochs"] == 2
            assert cfg["rollout_steps"] == 64
            assert cfg["sequence_chunk_len"] == 16
            assert cfg["num_minibatches"] == 4
            assert cfg["anneal_lr"] is False
            assert cfg["save_on_cancel"] is True

            for _ in range(50):
                state = server.training_manager.get_state()
                if state["status"] in ("completed", "error", "cancelled"):
                    break
                await asyncio.sleep(0.1)
            assert server.training_manager.get_state()["status"] == "completed"

    asyncio.run(_run_checks())


def test_save_on_cancel_persists_partial_checkpoint(tmp_path: Path) -> None:
    """Verify save_on_cancel=True saves a valid .pt checkpoint on cancellation."""
    direct_ckpt_dir = tmp_path / "direct_checkpoints"
    trainer = PPOTrainer(
        ppo_config=PPOConfig(
            total_timesteps=2048,
            num_envs=4,
            rollout_steps=32,
            sequence_chunk_len=8,
            num_minibatches=4,
            update_epochs=1,
            checkpoints_dir=direct_ckpt_dir,
        ),
        env_config=TrafficEnvConfig(max_steps=40),
    )

    cancel_after_first = threading.Event()

    def _cancel_on_first_update(metrics: dict[str, float]) -> None:
        if int(metrics["update"]) >= 1:
            cancel_after_first.set()

    _, history = trainer.train(
        model_type="mlp",
        verbose=False,
        on_update_callback=_cancel_on_first_update,
        cancel_event=cancel_after_first,
        save_checkpoint=True,
        checkpoint_filename="partial_mlp.pt",
        save_on_cancel=True,
    )
    assert len(history.updates) == 1
    partial_path = direct_ckpt_dir / "partial_mlp.pt"
    assert partial_path.exists()
    reloaded_policy, payload = load_policy_checkpoint(partial_path)
    assert reloaded_policy is not None
    assert payload["trained_timesteps"] == 128

    cancel_unsaved = threading.Event()

    def _cancel_unsaved_callback(metrics: dict[str, float]) -> None:
        if int(metrics["update"]) >= 1:
            cancel_unsaved.set()

    trainer.train(
        model_type="mlp",
        verbose=False,
        on_update_callback=_cancel_unsaved_callback,
        cancel_event=cancel_unsaved,
        save_checkpoint=True,
        checkpoint_filename="partial_unsaved.pt",
        save_on_cancel=False,
    )
    assert not (direct_ckpt_dir / "partial_unsaved.pt").exists()

    async def _run_web_cancel_checks() -> None:
        conn_path = tmp_path / "connectome.pt"
        ckpt_dir = tmp_path / "web_checkpoints"
        bench_dir = tmp_path / "web_benchmarks"
        artifact = _make_mock_artifact(num_neurons=100)
        artifact.save(conn_path)

        server = TrafficLabServer(
            frontend_dir=PROJECT_ROOT / "frontend",
            connectome_path=conn_path,
            checkpoints_dir=ckpt_dir,
            benchmarks_dir=bench_dir,
        )
        app = server.create_app()

        async with TestClient(TestServer(app)) as client:
            await client.post(
                "/api/simulation/control",
                json={"action": "set_running", "running": False},
            )
            async with (
                client.ws_connect("/ws/simulation") as sim_ws,
                client.ws_connect("/ws/training") as train_ws,
            ):
                await sim_ws.receive_json()
                await train_ws.receive_json()

                start_resp = await client.post(
                    "/api/training/start",
                    json={
                        "model_type": "mlp",
                        "total_timesteps": 65536,
                        "rollout_steps": 64,
                        "sequence_chunk_len": 16,
                        "num_minibatches": 4,
                        "update_epochs": 1,
                        "save_mode": "new_file",
                        "custom_filename": "cancelled_partial_mlp",
                        "save_on_cancel": True,
                    },
                )
                assert start_resp.status == 200

                cancelled_event: dict | None = None
                stop_sent = False
                for _ in range(40):
                    event = await asyncio.wait_for(
                        train_ws.receive_json(),
                        timeout=15.0,
                    )
                    if event.get("type") == "training_update" and not stop_sent:
                        stop_resp = await client.post("/api/training/stop")
                        assert stop_resp.status == 200
                        stop_sent = True
                    elif event.get("type") == "training_cancelled":
                        cancelled_event = event
                        break

                assert stop_sent
                assert cancelled_event is not None
                assert cancelled_event["status"] == "cancelled"
                assert (
                    cancelled_event["saved_checkpoint_file"]
                    == "cancelled_partial_mlp.pt"
                )
                assert (
                    cancelled_event["saved_agent_key"]
                    == "ckpt:cancelled_partial_mlp"
                )
                assert (ckpt_dir / "cancelled_partial_mlp.pt").exists()

                saw_overview = False
                saw_frame = False
                for _ in range(10):
                    sim_msg = await asyncio.wait_for(
                        sim_ws.receive_json(),
                        timeout=5.0,
                    )
                    if sim_msg.get("type") == "overview_updated":
                        if (
                            "ckpt:cancelled_partial_mlp"
                            in sim_msg["overview"]["checkpoints"]
                        ):
                            saw_overview = True
                    elif (
                        sim_msg.get("type") == "simulation_frame"
                        and sim_msg.get("agent_key") == "ckpt:cancelled_partial_mlp"
                    ):
                        saw_frame = True
                    if saw_overview and saw_frame:
                        break

                assert saw_overview
                assert saw_frame

    asyncio.run(_run_web_cancel_checks())
