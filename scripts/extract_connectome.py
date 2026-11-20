"""CLI entrypoint to extract and persist the Drosophila sub-connectome artifact."""

import argparse
from pathlib import Path

from fly_traffic.config import ConnectomeConfig
from fly_traffic.connectome.preprocessor import ConnectomePreprocessor


def parse_args() -> argparse.Namespace:
    """Parse command-line options for sub-connectome extraction."""
    parser = argparse.ArgumentParser(
        description="Extract CX/MB sub-connectome from FAFB v783 connections CSV."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=ConnectomeConfig.dataset_path,
        help="Path to connections_princeton.csv.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ConnectomeConfig.output_path,
        help="Output path for the serialized .pt connectome artifact.",
    )
    parser.add_argument(
        "--target-radius",
        type=float,
        default=ConnectomeConfig.target_spectral_radius,
        help="Target spectral radius for W_exc - W_inh.",
    )
    return parser.parse_args()


def main() -> None:
    """Run extraction, save artifact, and print factual verification metrics."""
    args = parse_args()
    config = ConnectomeConfig(
        dataset_path=args.dataset,
        output_path=args.output,
        target_spectral_radius=args.target_radius,
    )
    preprocessor = ConnectomePreprocessor(config=config)
    artifact = preprocessor.extract()
    artifact.save(config.output_path)

    print(f"Saved sub-connectome artifact to: {config.output_path}")
    for line in artifact.summary_lines():
        print(f"  {line}")


if __name__ == "__main__":
    main()
