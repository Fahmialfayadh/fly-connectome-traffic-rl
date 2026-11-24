#!/usr/bin/env python3
"""3D Traffic Intersection Simulation Server.

Directly powers the Three.js 3D viewer using the real Python TrafficIntersectionEnv,
real Drosophila connectome recurrent neural policies, Poisson arrival profiles,
and safety-interlocked signal controller from src/fly_traffic.
"""

import argparse
import asyncio
from contextlib import suppress
import json
from pathlib import Path
import sys
from typing import Any

from aiohttp import WSMsgType, web

# Ensure src/ is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fly_traffic.config import TrafficEnvConfig
from fly_traffic.web.simulation_service import SimulationService


class Traffic3DServer:
    """HTTP and WebSocket server bridging 3D visualization to real Python simulation."""

    def __init__(self, static_dir: Path, port: int = 8888) -> None:
        self.static_dir = static_dir
        self.port = port
        self.simulation_service = SimulationService(
            env_config=TrafficEnvConfig(),
        )
        self._clients: set[web.WebSocketResponse] = set()
        self._loop_task: asyncio.Task[None] | None = None

    def create_app(self) -> web.Application:
        app = web.Application()
        app.on_startup.append(self._on_startup)
        app.on_cleanup.append(self._on_cleanup)

        app.router.add_get("/", self._handle_index)
        app.router.add_get("/api/overview", self._handle_overview)
        app.router.add_post("/api/simulation/control", self._handle_control)
        app.router.add_get("/ws/simulation", self._handle_ws)

        # Serve static assets (js, css, shared 3D modules, vendor)
        app.router.add_static("/js", self.static_dir / "js")
        app.router.add_static("/css", self.static_dir / "css")
        sim3d_dir = PROJECT_ROOT / "frontend" / "js" / "sim3d"
        if sim3d_dir.exists():
            app.router.add_static("/sim3d", sim3d_dir)
        vendor_dir = PROJECT_ROOT / "frontend" / "vendor"
        if vendor_dir.exists():
            app.router.add_static("/vendor", vendor_dir)

        return app

    async def _on_startup(self, _app: web.Application) -> None:
        self._loop_task = asyncio.create_task(self._simulation_loop())

    async def _on_cleanup(self, _app: web.Application) -> None:
        if self._loop_task is not None:
            self._loop_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._loop_task

        for ws in list(self._clients):
            with suppress(Exception):
                await ws.close()

    async def _handle_index(self, _request: web.Request) -> web.Response:
        index_file = self.static_dir / "index.html"
        return web.Response(
            text=index_file.read_text(encoding="utf-8"),
            content_type="text/html",
            charset="utf-8",
            headers={"Cache-Control": "no-cache"},
        )

    async def _handle_overview(self, _request: web.Request) -> web.Response:
        overview = self.simulation_service.get_overview_payload()
        return web.json_response(overview)

    async def _handle_control(self, request: web.Request) -> web.Response:
        try:
            body = await request.json()
            if not isinstance(body, dict):
                raise ValueError("Payload must be a JSON object.")
            frame = self._apply_command(body)
        except Exception as exc:
            return web.json_response({"error": str(exc)}, status=400)

        await self._broadcast(frame)
        return web.json_response(frame)

    async def _handle_ws(self, request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse(heartbeat=20.0)
        await ws.prepare(request)
        self._clients.add(ws)

        try:
            # Send initial frame on connect
            await ws.send_json(self.simulation_service.get_frame_payload())
            async for message in ws:
                if message.type == WSMsgType.TEXT:
                    try:
                        payload = json.loads(message.data)
                        if isinstance(payload, dict):
                            frame = self._apply_command(payload)
                            await self._broadcast(frame)
                    except Exception as exc:
                        await ws.send_json({"type": "error", "error": str(exc)})
                elif message.type == WSMsgType.ERROR:
                    break
        finally:
            self._clients.discard(ws)

        return ws

    def _apply_command(self, body: dict[str, Any]) -> dict[str, Any]:
        action = str(body.get("action", "")).strip().lower()

        if action == "step":
            return self.simulation_service.step_once()
        if action == "reset":
            return self.simulation_service.reset_simulation(seed=body.get("seed", 42))
        if action == "set_running":
            return self.simulation_service.set_running(bool(body.get("running", True)))
        if action == "set_speed":
            return self.simulation_service.set_speed(float(body.get("steps_per_second", 2.0)))
        if action == "set_scenario":
            return self.simulation_service.set_scenario(str(body.get("scenario", "rush_hour")))
        if action == "set_safety_buffer":
            return self.simulation_service.set_safety_buffer(int(body.get("all_red_steps", 1)))
        if action == "set_continuous_mode":
            return self.simulation_service.set_continuous_mode(bool(body.get("enabled", True)))
        if action == "spawn_vehicles":
            lane_id = int(body.get("lane_id", 0))
            count = int(body.get("count", 1))
            return self.simulation_service.spawn_vehicles(lane_id=lane_id, count=count)
        if action == "spawn_random":
            count = int(body.get("count", 5))
            return self.simulation_service.spawn_random_vehicles(count=count)
        if action == "spawn_surge":
            count_per_lane = int(body.get("count_per_lane", 3))
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

        raise ValueError(f"Unknown command action '{action}'.")

    async def _simulation_loop(self) -> None:
        while True:
            sps = self.simulation_service.steps_per_second
            interval = 1.0 / max(sps, 0.5)
            await asyncio.sleep(interval)

            if self.simulation_service.is_running and self._clients:
                frame = self.simulation_service.step_once()
                await self._broadcast(frame)

    async def _broadcast(self, payload: dict[str, Any]) -> None:
        if not self._clients:
            return
        text_data = json.dumps(payload)
        dead: list[web.WebSocketResponse] = []
        for ws in list(self._clients):
            if ws.closed:
                dead.append(ws)
                continue
            try:
                await ws.send_str(text_data)
            except Exception:
                dead.append(ws)
        for d in dead:
            self._clients.discard(d)


def parse_args() -> argparse.Namespace:
    """Parse host and port arguments for the standalone 3D viewer server."""
    parser = argparse.ArgumentParser(
        description="Launch standalone 3D Traffic Intersection simulation server."
    )
    parser.add_argument(
        "port_pos",
        nargs="?",
        type=int,
        default=None,
        help="Optional positional TCP port (for backward compatibility).",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host interface to bind.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8888,
        help="TCP port for the HTTP and WebSocket server.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    port = args.port_pos if args.port_pos is not None else args.port
    static_dir = Path(__file__).resolve().parent
    server = Traffic3DServer(static_dir=static_dir, port=port)
    app = server.create_app()

    print("\n=======================================================")
    print("  3D Traffic Simulation (Connected to Python Env)")
    print(f"  Serving at: http://{args.host}:{port}")
    print("=======================================================\n")
    web.run_app(app, host=args.host, port=port, print=None)


if __name__ == "__main__":
    main()
