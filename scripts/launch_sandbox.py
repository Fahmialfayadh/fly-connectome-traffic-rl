"""CLI entrypoint for launching the 60 FPS Pygame interactive connectome sandbox."""

import argparse
from pathlib import Path

from fly_traffic.config import ConnectomeConfig, PPOConfig
from fly_traffic.ui.sandbox_app import TrafficConnectomeSandboxApp


def parse_args() -> argparse.Namespace:
    """Parse CLI options for interactive UI or headless preview rendering."""
    parser = argparse.ArgumentParser(
        description="Launch interactive 60 FPS Drosophila Connectome traffic sandbox."
    )
    parser.add_argument(
        "--connectome",
        type=Path,
        default=ConnectomeConfig.output_path,
        help="Path to fly_subconnectome.pt.",
    )
    parser.add_argument(
        "--checkpoints-dir",
        type=Path,
        default=PPOConfig.checkpoints_dir,
        help="Directory containing trained *_ppo.pt checkpoints.",
    )
    parser.add_argument(
        "--headless-screenshot",
        type=Path,
        default=None,
        help="Optional PNG output path to render headless without opening a display.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=45,
        help="Simulation steps to advance before saving headless screenshot.",
    )

    parser.add_argument(
        "--parallel-models",
        type=int,
        choices=[1, 2, 4],
        default=1,
        help="Maximum number of models trained concurrently.",
    )

    return parser.parse_args()


def main() -> None:
    """Initialize and launch the Pygame sandbox application."""
    args = parse_args()
    app = TrafficConnectomeSandboxApp(
        connectome_path=args.connectome,
        checkpoints_dir=args.checkpoints_dir,
    )
    if args.headless_screenshot is not None:
        saved = app.render_headless_screenshot(
            output_path=args.headless_screenshot,
            steps=args.steps,
        )
        print(f"Saved headless sandbox screenshot to: {saved}")
        return

    app.run()


if __name__ == "__main__":
    main()
