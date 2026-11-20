"""Build frontend/assets/brain/{fly_brain_meshes.glb, fly_brain_neurons.json}.

Raw inputs are expected in dataset/brain3d/ (see README "3D Brain Viewer" section):
    brain_mesh_v3.bin, JFRC2NP.surf.fw.zip, neuron_annotations.tsv
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fly_traffic.connectome.brain_assets import BrainAssetBuilder  # noqa: E402


def main() -> None:
    outputs = BrainAssetBuilder().build()
    for label, path in outputs.items():
        print(f"{label}: {path} ({path.stat().st_size / 1024:.1f} KiB)")


if __name__ == "__main__":
    main()
