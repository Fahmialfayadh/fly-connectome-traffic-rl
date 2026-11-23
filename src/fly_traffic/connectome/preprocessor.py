"""Polars + SciPy preprocessor for extracting the Drosophila CX/MB sub-connectome."""

from dataclasses import dataclass

import numpy as np
import polars as pl
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch

from fly_traffic.config import ConnectomeConfig
from fly_traffic.connectome.artifact import ConnectomeArtifact, ConnectomeMetadata


@dataclass(frozen=True)
class _NeuronRecord:
    """Internal sortable record describing a selected neuron's circuit annotation."""

    root_id: int
    circuit_group: str
    dominant_neuropil: str
    dominant_nt: str
    total_synapses: int


class ConnectomePreprocessor:
    """Extracts, separates by polarity, and spectrally normalizes a sub-connectome."""

    def __init__(self, config: ConnectomeConfig | None = None) -> None:
        self.config = config or ConnectomeConfig()

    def extract(self) -> ConnectomeArtifact:
        """Run the full extraction pipeline from connections CSV to sparse artifact."""
        if not self.config.dataset_path.exists():
            raise FileNotFoundError(
                f"Connectome dataset not found at: {self.config.dataset_path}"
            )

        lazy_edge_table = pl.scan_csv(self.config.dataset_path)

        central_complex_ids = self._select_top_neurons(
            lazy_edge_table=lazy_edge_table,
            target_neuropils=self.config.cx_neuropils, 
            neuron_quota=self.config.cx_neuron_quota, # 1800
            excluded_root_ids=set(),
        )

        central_complex_id_set = set(central_complex_ids)

        mushroom_body_ids = self._select_top_neurons(
            lazy_edge_table=lazy_edge_table,
            target_neuropils=self.config.mb_neuropils,
            neuron_quota=self.config.mb_neuron_quota,
            excluded_root_ids=central_complex_id_set,
        )
        mushroom_body_id_set = set(mushroom_body_ids)
        selected_root_ids = central_complex_ids + mushroom_body_ids

        # Induce subgraph across all neuropils so CX <-> MB bridges are preserved.
        induced_edge_table = (
            lazy_edge_table.filter(
                pl.col("pre_root_id").is_in(selected_root_ids)
                & pl.col("post_root_id").is_in(selected_root_ids)
            )
            .collect()
        )

        sorted_neuron_records = self._annotate_and_sort_neurons(
            induced_edge_table=induced_edge_table,
            selected_root_ids=selected_root_ids,
            central_complex_id_set=central_complex_id_set,
        )
        ordered_root_ids = [record.root_id for record in sorted_neuron_records]
        root_id_to_index = {
            root_id: matrix_index
            for matrix_index, root_id in enumerate(ordered_root_ids)
        }
        num_neurons = len(ordered_root_ids)

        excitatory_edge_table = induced_edge_table.filter(
            pl.col("nt_type").is_in(self.config.excitatory_nts)
        )
        inhibitory_edge_table = induced_edge_table.filter(
            pl.col("nt_type").is_in(self.config.inhibitory_nts)
        )

        raw_excitatory_matrix = self._build_polarity_csr(
            edge_table=excitatory_edge_table,
            root_id_to_index=root_id_to_index,
            num_neurons=num_neurons,
        )
        raw_inhibitory_matrix = self._build_polarity_csr(
            edge_table=inhibitory_edge_table,
            root_id_to_index=root_id_to_index,
            num_neurons=num_neurons,
        )

        # Normalize spectral radius of net biological matrix (W_exc - W_inh).
        raw_net_matrix = (raw_excitatory_matrix - raw_inhibitory_matrix).astype(np.float64)
        raw_spectral_radius = self._compute_spectral_radius(raw_net_matrix)
        scaling_factor = (
            self.config.target_spectral_radius / raw_spectral_radius
            if raw_spectral_radius > 0.0
            else 1.0
        )

        scaled_excitatory_matrix = (raw_excitatory_matrix * scaling_factor).astype(np.float32)
        scaled_inhibitory_matrix = (raw_inhibitory_matrix * scaling_factor).astype(np.float32)
        scaled_spectral_radius = self._compute_spectral_radius(
            (scaled_excitatory_matrix - scaled_inhibitory_matrix).astype(np.float64)
        )

        total_possible_edges = num_neurons * num_neurons
        combined_nonzero_edges = scaled_excitatory_matrix.nnz + scaled_inhibitory_matrix.nnz
        sparsity = 1.0 - (combined_nonzero_edges / total_possible_edges)
        cross_region_edges = self._count_cross_region_edges(
            induced_edge_table=induced_edge_table,
            central_complex_id_set=central_complex_id_set,
            mushroom_body_id_set=mushroom_body_id_set,
        )

        metadata = ConnectomeMetadata(
            num_neurons=num_neurons,
            cx_count=len(central_complex_ids),
            mb_count=len(mushroom_body_ids),
            exc_edge_count=int(scaled_excitatory_matrix.nnz),
            inh_edge_count=int(scaled_inhibitory_matrix.nnz),
            cross_region_edges=cross_region_edges,
            sparsity=float(sparsity),
            raw_spectral_radius=float(raw_spectral_radius),
            scaled_spectral_radius=float(scaled_spectral_radius),
        )

        excitatory_indices, excitatory_values = self._csr_to_coo_tensors(scaled_excitatory_matrix)
        inhibitory_indices, inhibitory_values = self._csr_to_coo_tensors(scaled_inhibitory_matrix)

        return ConnectomeArtifact(
            w_exc_indices=excitatory_indices,
            w_exc_values=excitatory_values,
            w_inh_indices=inhibitory_indices,
            w_inh_values=inhibitory_values,
            neuron_ids=ordered_root_ids,
            circuit_groups=[record.circuit_group for record in sorted_neuron_records],
            dominant_neuropils=[record.dominant_neuropil for record in sorted_neuron_records],
            dominant_nts=[record.dominant_nt for record in sorted_neuron_records],
            metadata=metadata,
        )

    def _select_top_neurons(
        self,
        lazy_edge_table: pl.LazyFrame,
        target_neuropils: tuple[str, ...],
        neuron_quota: int,
        excluded_root_ids: set[int],
    ) -> list[int]:
        """Select top-k neurons within target neuropils ranked by incident synapses."""
        region_edges = lazy_edge_table.filter(pl.col("neuropil").is_in(target_neuropils)).collect()

        outgoing_synapses = region_edges.group_by("pre_root_id").agg(
            pl.col("syn_count").sum().alias("synapse_sum")
        ).rename({"pre_root_id": "root_id"})

        incoming_synapses = region_edges.group_by("post_root_id").agg(
            pl.col("syn_count").sum().alias("synapse_sum")
        ).rename({"post_root_id": "root_id"})

        ranked_neurons = (
            pl.concat([outgoing_synapses, incoming_synapses])
            .group_by("root_id")
            .agg(pl.col("synapse_sum").sum().alias("total_synapses"))
            .sort(["total_synapses", "root_id"], descending=[True, False])
        )

        selected_ids: list[int] = []
        for raw_root_id in ranked_neurons["root_id"].to_list():
            neuron_root_id = int(raw_root_id)
            if neuron_root_id in excluded_root_ids:
                continue

            selected_ids.append(neuron_root_id)
            if len(selected_ids) >= neuron_quota:
                break

        return selected_ids

    def _annotate_and_sort_neurons(
        self,
        induced_edge_table: pl.DataFrame,
        selected_root_ids: list[int],
        central_complex_id_set: set[int],
    ) -> list[_NeuronRecord]:
        """Compute anatomical labels and sort neurons into contiguous circuit bands."""
        target_neuropil_set = set(self.config.cx_neuropils) | set(self.config.mb_neuropils)

        dominant_neuropil_frame = (
            pl.concat(
                [
                    induced_edge_table.select(
                        pl.col("pre_root_id").alias("root_id"),
                        pl.col("neuropil"),
                        pl.col("syn_count"),
                    ),
                    induced_edge_table.select(
                        pl.col("post_root_id").alias("root_id"),
                        pl.col("neuropil"),
                        pl.col("syn_count"),
                    ),
                ]
            )
            .filter(pl.col("neuropil").is_in(list(target_neuropil_set)))
            .group_by(["root_id", "neuropil"])
            .agg(pl.col("syn_count").sum().alias("synapse_sum"))
            .sort(["root_id", "synapse_sum", "neuropil"], descending=[False, True, False])
            .unique(subset=["root_id"], keep="first")
        )
        dominant_neuropil_map = {
            int(row["root_id"]): str(row["neuropil"])
            for row in dominant_neuropil_frame.iter_rows(named=True)
        }

        outgoing_nt_frame = (
            induced_edge_table.group_by(["pre_root_id", "nt_type"])
            .agg(pl.col("syn_count").sum().alias("synapse_sum"))
            .sort(
                ["pre_root_id", "synapse_sum", "nt_type"],
                descending=[False, True, False],
            )
            .unique(subset=["pre_root_id"], keep="first")
        )
        dominant_nt_map = {
            int(row["pre_root_id"]): str(row["nt_type"])
            for row in outgoing_nt_frame.iter_rows(named=True)
        }

        total_synapse_frame = (
            pl.concat(
                [
                    induced_edge_table.select(
                        pl.col("pre_root_id").alias("root_id"),
                        pl.col("syn_count"),
                    ),
                    induced_edge_table.select(
                        pl.col("post_root_id").alias("root_id"),
                        pl.col("syn_count"),
                    ),
                ]
            )
            .group_by("root_id")
            .agg(pl.col("syn_count").sum().alias("total_synapses"))
        )
        total_synapse_map = {
            int(row["root_id"]): int(row["total_synapses"])
            for row in total_synapse_frame.iter_rows(named=True)
        }

        neuron_records: list[_NeuronRecord] = []
        for neuron_root_id in selected_root_ids:
            circuit_group = "CX" if neuron_root_id in central_complex_id_set else "MB"
            neuron_records.append(
                _NeuronRecord(
                    root_id=neuron_root_id,
                    circuit_group=circuit_group,
                    dominant_neuropil=dominant_neuropil_map.get(neuron_root_id, circuit_group),
                    dominant_nt=dominant_nt_map.get(neuron_root_id, "UNKNOWN"),
                    total_synapses=total_synapse_map.get(neuron_root_id, 0),
                )
            )

        neuron_records.sort(
            key=lambda record: (
                0 if record.circuit_group == "CX" else 1,
                record.dominant_neuropil,
                record.dominant_nt,
                -record.total_synapses,
                record.root_id,
            )
        )
        return neuron_records

    @staticmethod
    def _build_polarity_csr(
        edge_table: pl.DataFrame,
        root_id_to_index: dict[int, int],
        num_neurons: int,
    ) -> sp.csr_matrix:
        """Build W[post_idx, pre_idx] = log1p(sum_syn_count) in SciPy CSR format."""
        if edge_table.is_empty():
            return sp.csr_matrix((num_neurons, num_neurons), dtype=np.float32)

        grouped_edges = edge_table.group_by(["post_root_id", "pre_root_id"]).agg(
            pl.col("syn_count").sum().alias("synapse_sum")
        )
        post_indices = np.fromiter(
            (root_id_to_index[int(nid)] for nid in grouped_edges["post_root_id"].to_list()),
            dtype=np.int64,
            count=len(grouped_edges),
        )
        pre_indices = np.fromiter(
            (root_id_to_index[int(nid)] for nid in grouped_edges["pre_root_id"].to_list()),
            dtype=np.int64,
            count=len(grouped_edges),
        )
        synapse_weights = np.log1p(grouped_edges["synapse_sum"].to_numpy().astype(np.float32))

        sparse_matrix = sp.csr_matrix(
            (synapse_weights, (post_indices, pre_indices)),
            shape=(num_neurons, num_neurons),
            dtype=np.float32,
        )
        sparse_matrix.sum_duplicates()
        return sparse_matrix

    def _compute_spectral_radius(self, matrix: sp.csr_matrix) -> float:
        """Compute largest-magnitude eigenvalue via Arnoldi with power-iteration fallback."""
        if matrix.nnz == 0:
            return 0.0

        generator = np.random.default_rng(self.config.power_iteration_seed)
        initial_vector = generator.standard_normal(matrix.shape[0]).astype(np.float64)

        try:
            eigenvalues = spla.eigs(
                matrix,
                k=1,
                which="LM",
                v0=initial_vector,
                maxiter=1000,
                return_eigenvectors=False,
            )
            return float(np.abs(eigenvalues[0]))
        except spla.ArpackNoConvergence:
            return self._power_iteration_radius(matrix, initial_vector)

    @staticmethod
    def _power_iteration_radius(
        matrix: sp.csr_matrix,
        initial_vector: np.ndarray,
        iterations: int = 200,
    ) -> float:
        """Fallback Rayleigh-norm estimate when ARPACK fails to converge."""
        unit_vector = initial_vector / max(float(np.linalg.norm(initial_vector)), 1e-12)
        radius_estimate = 0.0

        for _ in range(iterations):
            next_vector = matrix @ unit_vector
            vector_norm = float(np.linalg.norm(next_vector))
            if vector_norm < 1e-12:
                return 0.0

            radius_estimate = vector_norm
            unit_vector = next_vector / vector_norm

        return radius_estimate

    @staticmethod
    def _count_cross_region_edges(
        induced_edge_table: pl.DataFrame,
        central_complex_id_set: set[int],
        mushroom_body_id_set: set[int],
    ) -> int:
        """Count directed synaptic rows linking Central Complex and Mushroom Body."""
        cx_list = list(central_complex_id_set)
        mb_list = list(mushroom_body_id_set)
        cross_region_frame = induced_edge_table.filter(
            (pl.col("pre_root_id").is_in(cx_list) & pl.col("post_root_id").is_in(mb_list))
            | (pl.col("pre_root_id").is_in(mb_list) & pl.col("post_root_id").is_in(cx_list))
        )
        return int(len(cross_region_frame))

    @staticmethod
    def _csr_to_coo_tensors(matrix: sp.csr_matrix) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert SciPy CSR matrix into coalesced COO index and value tensors."""
        coo_matrix = matrix.tocoo()
        indices = np.vstack([coo_matrix.row, coo_matrix.col]).astype(np.int64)
        values = coo_matrix.data.astype(np.float32)
        return torch.from_numpy(indices), torch.from_numpy(values)
