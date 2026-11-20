"""Headless Pygame smoke test for sandbox rendering and interactive callbacks."""

import os
from pathlib import Path

from PIL import Image
import numpy as np

from fly_traffic.ui.sandbox_app import TrafficConnectomeSandboxApp
from tests.test_policies import _make_mock_artifact


def test_sandbox_headless_render_and_callbacks(tmp_path: Path, monkeypatch) -> None:
    """Verify headless Pygame rendering produces a valid non-blank 1280x760 image."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    os.environ["SDL_VIDEODRIVER"] = "dummy"

    conn_path = tmp_path / "connectome.pt"
    artifact = _make_mock_artifact(num_neurons=2500)
    artifact.save(conn_path)

    app = TrafficConnectomeSandboxApp(
        connectome_path=conn_path,
        checkpoints_dir=tmp_path / "missing_checkpoints",
    )

    # Exercise callbacks
    app._on_g_exc_change(1.75)
    app._on_g_inh_change(0.45)
    app._on_lesion_cx()
    assert app.fly_policy is not None
    assert app.fly_policy.cell.active_neuron_count() == 2050
    app._on_reset_brain()
    assert app.fly_policy.cell.active_neuron_count() == 2500

    screenshot = tmp_path / "sandbox_test.png"
    saved_path = app.render_headless_screenshot(screenshot, steps=15)
    assert saved_path.exists()

    img = Image.open(saved_path)
    assert img.size == (1280, 760)
    arr = np.asarray(img)
    assert arr.std() > 5.0
