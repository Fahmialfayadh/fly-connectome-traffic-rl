"""Serializable sub-connectome artifact container and sparse tensor converter."""

from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass(frozen=True)
class ConnectomeMetadata:
    """Summary statistics recorded during sub-connectome extraction."""

    num_neurons: int
    cx_count: int
    mb_count: int
    exc_edge_count: int
    inh_edge_count: int
    cross_region_edges: int
    sparsity: float
    raw_spectral_radius: float
    scaled_spectral_radius: float

    def to_dict(self) -> dict[str, int | float]:
        """Serialize metadata into plain Python primitives for safe checkpointing."""
        return {
            "num_neurons": int(self.num_neurons),
            "cx_count": int(self.cx_count),
            "mb_count": int(self.mb_count),
            "exc_edge_count": int(self.exc_edge_count),
            "inh_edge_count": int(self.inh_edge_count),
            "cross_region_edges": int(self.cross_region_edges),
            "sparsity": float(self.sparsity),
            "raw_spectral_radius": float(self.raw_spectral_radius),
            "scaled_spectral_radius": float(self.scaled_spectral_radius),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, int | float]) -> "ConnectomeMetadata":
        """Reconstruct metadata from a serialized dictionary."""
        return cls(
            num_neurons=int(payload["num_neurons"]),
            cx_count=int(payload["cx_count"]),
            mb_count=int(payload["mb_count"]),
            exc_edge_count=int(payload["exc_edge_count"]),
            inh_edge_count=int(payload["inh_edge_count"]),
            cross_region_edges=int(payload["cross_region_edges"]),
            sparsity=float(payload["sparsity"]),
            raw_spectral_radius=float(payload["raw_spectral_radius"]),
            scaled_spectral_radius=float(payload["scaled_spectral_radius"]),
        )


@dataclass
class ConnectomeArtifact:
    """Extracted biological excitatory/inhibitory sparse connectome matrices."""

    w_exc_indices: torch.Tensor
    w_exc_values: torch.Tensor
    w_inh_indices: torch.Tensor
    w_inh_values: torch.Tensor
    neuron_ids: list[int]
    circuit_groups: list[str]
    dominant_neuropils: list[str]
    dominant_nts: list[str]
    metadata: ConnectomeMetadata

    def save(self, path: Path) -> None:
        """Persist the artifact as weights_only-safe tensors and primitive metadata."""
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "w_exc_indices": self.w_exc_indices.cpu().to(torch.int64),
            "w_exc_values": self.w_exc_values.cpu().to(torch.float32),
            "w_inh_indices": self.w_inh_indices.cpu().to(torch.int64),
            "w_inh_values": self.w_inh_values.cpu().to(torch.float32),
            "neuron_ids": [int(nid) for nid in self.neuron_ids],
            "circuit_groups": list(self.circuit_groups),
            "dominant_neuropils": list(self.dominant_neuropils),
            "dominant_nts": list(self.dominant_nts),
            "metadata": self.metadata.to_dict(),
        }
        torch.save(payload, path)

    @classmethod
    def load(cls, path: Path) -> "ConnectomeArtifact":
        """Load a saved sub-connectome artifact cleanly with weights_only=True."""
        payload = torch.load(path, map_location="cpu", weights_only=True)
        return cls(
            w_exc_indices=payload["w_exc_indices"].to(torch.int64),
            w_exc_values=payload["w_exc_values"].to(torch.float32),
            w_inh_indices=payload["w_inh_indices"].to(torch.int64),
            w_inh_values=payload["w_inh_values"].to(torch.float32),
            neuron_ids=[int(nid) for nid in payload["neuron_ids"]],
            circuit_groups=list(payload["circuit_groups"]),
            dominant_neuropils=list(payload["dominant_neuropils"]),
            dominant_nts=list(payload["dominant_nts"]),
            metadata=ConnectomeMetadata.from_dict(payload["metadata"]),
        )

    def to_sparse_csr_tensors(
        self,
        device: torch.device | str = "cpu",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build coalesced PyTorch sparse CSR matrices for fast CPU forward passes."""
        target_device = torch.device(device)
        num_neurons = self.metadata.num_neurons
        shape = (num_neurons, num_neurons)

        torch.sparse.check_sparse_tensor_invariants.disable()
        w_exc_coo = torch.sparse_coo_tensor(
            self.w_exc_indices.to(target_device),
            self.w_exc_values.to(target_device),
            size=shape,
            dtype=torch.float32,
        ).coalesce()
        w_inh_coo = torch.sparse_coo_tensor(
            self.w_inh_indices.to(target_device),
            self.w_inh_values.to(target_device),
            size=shape,
            dtype=torch.float32,
        ).coalesce()

        return w_exc_coo.to_sparse_csr(), w_inh_coo.to_sparse_csr()

    def summary_lines(self) -> list[str]:
        """Return factual summary lines describing extracted graph properties."""
        meta = self.metadata
        return [
            f"Neurons: {meta.num_neurons} (CX={meta.cx_count}, MB={meta.mb_count})",
            (
                f"Directed Edges: W_exc={meta.exc_edge_count}, "
                f"W_inh={meta.inh_edge_count}, CX<->MB={meta.cross_region_edges}"
            ),
            f"Combined Sparsity: {meta.sparsity:.4%}",
            (
                f"Spectral Radius: raw={meta.raw_spectral_radius:.4f}, "
                f"scaled={meta.scaled_spectral_radius:.4f}"
            ),
        ]
