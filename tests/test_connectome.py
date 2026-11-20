"""Unit tests for sub-connectome extraction, spectral scaling, and serialization."""

from pathlib import Path

import numpy as np
import polars as pl
import torch

from fly_traffic.config import ConnectomeConfig
from fly_traffic.connectome.artifact import ConnectomeArtifact
from fly_traffic.connectome.preprocessor import ConnectomePreprocessor


def _create_synthetic_connections_csv(path: Path) -> None:
    rng = np.random.default_rng(7)
    cx_nodes = list(range(1000, 1040))
    mb_nodes = list(range(2000, 2020))
    all_nodes = cx_nodes + mb_nodes

    rows: list[dict[str, object]] = []
    for pre in cx_nodes:
        targets = rng.choice(all_nodes, size=4, replace=False)
        for post in targets:
            nt = "ACH" if rng.random() < 0.55 else ("GABA" if rng.random() < 0.5 else "GLUT")
            rows.append(
                {
                    "pre_root_id": int(pre),
                    "post_root_id": int(post),
                    "neuropil": "EB" if post in cx_nodes else "MB_ML_L",
                    "syn_count": int(rng.integers(5, 80)),
                    "nt_type": nt,
                }
            )

    for pre in mb_nodes:
        targets = rng.choice(all_nodes, size=3, replace=False)
        for post in targets:
            nt = "ACH" if rng.random() < 0.5 else "GABA"
            rows.append(
                {
                    "pre_root_id": int(pre),
                    "post_root_id": int(post),
                    "neuropil": "MB_CA_L",
                    "syn_count": int(rng.integers(5, 60)),
                    "nt_type": nt,
                }
            )

    pl.DataFrame(rows).write_csv(path)


def test_connectome_extraction_and_roundtrip(tmp_path: Path) -> None:
    """Verify quota extraction, E/I positivity, spectral radius, and save/load."""
    csv_path = tmp_path / "connections.csv"
    out_path = tmp_path / "subconnectome.pt"
    _create_synthetic_connections_csv(csv_path)

    cfg = ConnectomeConfig(
        dataset_path=csv_path,
        output_path=out_path,
        cx_neuron_quota=30,
        mb_neuron_quota=15,
        target_spectral_radius=0.95,
    )
    artifact = ConnectomePreprocessor(cfg).extract()

    assert artifact.metadata.num_neurons == 45
    assert artifact.metadata.cx_count == 30
    assert artifact.metadata.mb_count == 15
    assert artifact.metadata.sparsity > 0.80
    assert torch.all(artifact.w_exc_values > 0.0)
    assert torch.all(artifact.w_inh_values > 0.0)
    assert abs(artifact.metadata.scaled_spectral_radius - 0.95) < 0.02

    artifact.save(out_path)
    reloaded = ConnectomeArtifact.load(out_path)

    assert reloaded.metadata == artifact.metadata
    assert reloaded.neuron_ids == artifact.neuron_ids
    assert torch.equal(reloaded.w_exc_indices, artifact.w_exc_indices)
    assert torch.allclose(reloaded.w_inh_values, artifact.w_inh_values)

    w_exc_csr, w_inh_csr = reloaded.to_sparse_csr_tensors()
    assert w_exc_csr.shape == (45, 45)
    assert w_inh_csr.shape == (45, 45)
