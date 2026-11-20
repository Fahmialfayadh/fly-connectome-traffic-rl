"""CLI entrypoint launching the aiohttp Light Instrument Lab web server."""

import argparse
from pathlib import Path

from aiohttp import web

from fly_traffic.config import (
    ConnectomeConfig,
    DEFAULT_ARTIFACTS_DIR,
    PPOConfig,
    PROJECT_ROOT,
)
from fly_traffic.web.server import TrafficLabServer


def parse_args() -> argparse.Namespace:
    """Parse CLI host, port, and artifact paths for the web server."""
    parser = argparse.ArgumentParser(
        description="Launch the Drosophila Connectome Traffic Control web interface."
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
        default=8080,
        help="TCP port for the HTTP and WebSocket server.",
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
        help="Directory containing trained *_ppo.pt checkpoints.",
    )
    parser.add_argument(
        "--benchmarks-dir",
        type=Path,
        default=DEFAULT_ARTIFACTS_DIR / "benchmarks",
        help="Directory containing benchmark JSON and plots.",
    )
    return parser.parse_args()


def main() -> None:
    """Start the aiohttp web server."""
    args = parse_args()
    server = TrafficLabServer(
        frontend_dir=PROJECT_ROOT / "frontend",
        connectome_path=args.connectome,
        checkpoints_dir=args.checkpoints_dir,
        benchmarks_dir=args.benchmarks_dir,
    )
    app = server.create_app()
    print(f"Serving Drosophila Connectome Traffic Lab at http://{args.host}:{args.port}")
    web.run_app(app, host=args.host, port=args.port, print=None)


if __name__ == "__main__":
    main()
