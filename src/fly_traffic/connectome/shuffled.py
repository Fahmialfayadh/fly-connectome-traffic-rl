"""Controlled null-model generator: global random edge rewiring of the FAFB connectome.

Preserves exact neuron count (2,500), exact excitatory/inhibitory edge counts and
weight distributions, exact matrix sparsity, and target spectral radius (0.95),
while destroying biological modularity (CX/MB structure and cross-region bridges).
"""

from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch

from fly_traffic.config import ConnectomeConfig
from fly_traffic.connectome.artifact import ConnectomeArtifact, ConnectomeMetadata
from fly_traffic.connectome.preprocessor import ConnectomePreprocessor


def default_shuffled_connectome_path(connectome_path: Path | None = None) -> Path:
    """Return sibling artifact path for the shuffled null-model connectome."""
    base = connectome_path or ConnectomeConfig.output_path
    return base.with_name(f"{base.stem}_shuffled{base.suffix}")


def _rewire_edges_globally(
    indices: torch.Tensor,
    values: torch.Tensor,
    num_neurons: int,
    rng: np.random.Generator,
) -> sp.csr_matrix:
    """Assign coalesced synaptic weights to uniformly sampled unique (post, pre) pairs."""
    if int(values.numel()) == 0:
        return sp.csr_matrix((num_neurons, num_neurons), dtype=np.float32)

    # Coalesce duplicate (row, col) pairs first so edge_count <= num_neurons^2.
    src_rows = indices[0].detach().cpu().numpy().astype(np.int64)
    src_cols = indices[1].detach().cpu().numpy().astype(np.int64)
    src_vals = values.detach().cpu().numpy().astype(np.float32)
    coalesced = sp.coo_matrix(
        (src_vals, (src_rows, src_cols)),
        shape=(num_neurons, num_neurons),
        dtype=np.float32,
    ).tocsr()
    coalesced.sum_duplicates()
    coalesced.eliminate_zeros()

    edge_count = int(coalesced.nnz)
    if edge_count == 0:
        return sp.csr_matrix((num_neurons, num_neurons), dtype=np.float32)

    total_slots = num_neurons * num_neurons
    flat_indices = rng.choice(total_slots, size=edge_count, replace=False)
    rows = (flat_indices // num_neurons).astype(np.int64)
    cols = (flat_indices % num_neurons).astype(np.int64)
    weights = coalesced.data.astype(np.float32, copy=True)

    return sp.coo_matrix(
        (weights, (rows, cols)),
        shape=(num_neurons, num_neurons),
        dtype=np.float32,
    ).tocsr()


def create_shuffled_connectome_artifact(
    source_artifact: ConnectomeArtifact,
    seed: int = 42,
    target_spectral_radius: float = 0.95,
) -> ConnectomeArtifact:
    """Create a topology-destroyed null-model artifact preserving E/I weight distributions."""
    num_neurons = source_artifact.metadata.num_neurons
    rng = np.random.default_rng(seed)

    raw_exc = _rewire_edges_globally(
        source_artifact.w_exc_indices, source_artifact.w_exc_values, num_neurons, rng
    )
    raw_inh = _rewire_edges_globally(
        source_artifact.w_inh_indices, source_artifact.w_inh_values, num_neurons, rng
    )

    preprocessor = ConnectomePreprocessor(
        ConnectomeConfig(target_spectral_radius=target_spectral_radius, power_iteration_seed=seed)
    )
    raw_net = (raw_exc - raw_inh).astype(np.float64)
    raw_rho = preprocessor._compute_spectral_radius(raw_net)
    scale = target_spectral_radius / raw_rho if raw_rho > 0.0 else 1.0

    scaled_exc = (raw_exc * scale).astype(np.float32)
    scaled_inh = (raw_inh * scale).astype(np.float32)
    scaled_rho = preprocessor._compute_spectral_radius(
        (scaled_exc - scaled_inh).astype(np.float64)
    )

    total_possible = num_neurons * num_neurons
    combined_nnz = int(scaled_exc.nnz + scaled_inh.nnz)
    sparsity = 1.0 - (combined_nnz / total_possible)

    cx_mask = np.array(
        [grp == "CX" for grp in source_artifact.circuit_groups],
        dtype=bool,
    )
    exc_coo = scaled_exc.tocoo()
    inh_coo = scaled_inh.tocoo()
    cross_exc = int(np.sum(cx_mask[exc_coo.row] != cx_mask[exc_coo.col]))
    cross_inh = int(np.sum(cx_mask[inh_coo.row] != cx_mask[inh_coo.col]))

    metadata = ConnectomeMetadata(
        num_neurons=num_neurons,
        cx_count=source_artifact.metadata.cx_count,
        mb_count=source_artifact.metadata.mb_count,
        exc_edge_count=int(scaled_exc.nnz),
        inh_edge_count=int(scaled_inh.nnz),
        cross_region_edges=cross_exc + cross_inh,
        sparsity=float(sparsity),
        raw_spectral_radius=float(raw_rho),
        scaled_spectral_radius=float(scaled_rho),
    )

    exc_indices = torch.tensor(
        np.vstack([exc_coo.row, exc_coo.col]),
        dtype=torch.int64,
    )
    exc_values = torch.tensor(exc_coo.data, dtype=torch.float32)
    inh_indices = torch.tensor(
        np.vstack([inh_coo.row, inh_coo.col]),
        dtype=torch.int64,
    )
    inh_values = torch.tensor(inh_coo.data, dtype=torch.float32)

    return ConnectomeArtifact(
        w_exc_indices=exc_indices,
        w_exc_values=exc_values,
        w_inh_indices=inh_indices,
        w_inh_values=inh_values,
        neuron_ids=list(source_artifact.neuron_ids),
        circuit_groups=list(source_artifact.circuit_groups),
        dominant_neuropils=list(source_artifact.dominant_neuropils),
        dominant_nts=list(source_artifact.dominant_nts),
        metadata=metadata,
    )


def load_or_create_shuffled_connectome(
    connectome_path: Path | None = None,
    shuffled_path: Path | None = None,
    seed: int = 42,
    target_spectral_radius: float = 0.95,
) -> ConnectomeArtifact:
    """Load cached shuffled null-model artifact or generate & persist it deterministically."""
    base_path = connectome_path or ConnectomeConfig.output_path
    target_path = shuffled_path or default_shuffled_connectome_path(base_path)
    if target_path.exists():
        return ConnectomeArtifact.load(target_path)

    source_artifact = ConnectomeArtifact.load(base_path)
    shuffled = create_shuffled_connectome_artifact(
        source_artifact=source_artifact,
        seed=seed,
        target_spectral_radius=target_spectral_radius,
    )
    shuffled.save(target_path)
    return shuffled
