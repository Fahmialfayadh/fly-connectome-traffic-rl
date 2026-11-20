"""Build web-ready 3D anatomy assets for the live Drosophila brain viewer.

Sources (all FlyWire / FAFB14.1 space, nanometres):
- Whole-brain neuropil surface: ``gs://flywire_neuropil_meshes/whole_neuropil/brain_mesh_v3``
  (Neuroglancer legacy mesh fragment ``1:0:0``).
- Per-neuropil surfaces: ``JFRC2NP.surf.fw.zip`` shipped with ``fafbseg-py``
  (JFRC2 neuropil meshes transformed to FlyWire space, zenodo.10567).
- Neuron soma coordinates and cell types: FlyWire ``Supplemental_file1_neuron_annotations.tsv``
  (Schlegel et al. 2024, voxel units 4 x 4 x 40 nm, root ids v783).

Coordinate convention of the exported scene:
    +X = fly left, +Y = dorsal, +Z = anterior (camera on +Z sees the fly's face).
FAFB voxel space is anatomically mirror-inverted (x = right, y = ventral, z = posterior),
so the export applies a point reflection and re-orients triangle winding.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import io
import json
import math
from pathlib import Path
import zipfile

import numpy as np
import polars as pl
import torch
import trimesh

from fly_traffic.config import PROJECT_ROOT
from fly_traffic.connectome.artifact import ConnectomeArtifact

CX_NEUROPILS: tuple[str, ...] = ("PB", "EB", "FB", "NO")
MB_NEUROPILS: tuple[str, ...] = (
    "MB_CA_L",
    "MB_CA_R",
    "MB_PED_L",
    "MB_PED_R",
    "MB_VL_L",
    "MB_VL_R",
    "MB_ML_L",
    "MB_ML_R",
)
VOXEL_SIZE_NM = np.array([4.0, 4.0, 40.0], dtype=np.float64)
SCENE_WIDTH_UNITS = 10.0


@dataclass(frozen=True)
class BrainAssetConfig:
    """Input and output locations for the 3D brain asset build."""

    raw_dir: Path = PROJECT_ROOT / "dataset" / "brain3d"
    connectome_path: Path = PROJECT_ROOT / "artifacts" / "fly_subconnectome.pt"
    output_dir: Path = PROJECT_ROOT / "frontend" / "assets" / "brain"
    brain_mesh_file: str = "brain_mesh_v3.bin"
    neuropil_zip_file: str = "JFRC2NP.surf.fw.zip"
    annotations_file: str = "neuron_annotations.tsv"
    top_edge_count: int = 3000
    seed: int = 7
    sources: dict[str, str] = field(
        default_factory=lambda: {
            "brain_mesh": "https://storage.googleapis.com/flywire_neuropil_meshes/whole_neuropil/brain_mesh_v3/mesh/1:0:0",
            "neuropil_meshes": "https://github.com/navis-org/fafbseg-py/raw/master/fafbseg/data/JFRC2NP.surf.fw.zip",
            "neuron_annotations": "https://raw.githubusercontent.com/flyconnectome/flywire_annotations/main/supplemental_files/Supplemental_file1_neuron_annotations.tsv",
        }
    )


def load_legacy_ng_mesh(raw_bytes: bytes) -> trimesh.Trimesh:
    """Decode a Neuroglancer legacy mesh fragment (uint32 nv, f32 verts, u32 faces)."""
    vertex_count = int(np.frombuffer(raw_bytes[:4], dtype=np.uint32)[0])
    vertex_end = 4 + vertex_count * 12
    vertices = np.frombuffer(raw_bytes[4:vertex_end], dtype=np.float32).reshape(-1, 3)
    faces = np.frombuffer(raw_bytes[vertex_end:], dtype=np.uint32).reshape(-1, 3)
    return trimesh.Trimesh(vertices=vertices.astype(np.float64), faces=faces, process=True)


def generalized_winding_number(points: np.ndarray, mesh: trimesh.Trimesh) -> np.ndarray:
    """Robust inside test (|w| > 0.5) that tolerates non-watertight neuropil meshes."""
    triangles = mesh.triangles
    output = np.zeros(len(points), dtype=np.float64)
    for start in range(0, len(points), 256):
        chunk = points[start : start + 256]
        a = triangles[None, :, 0, :] - chunk[:, None, :]
        b = triangles[None, :, 1, :] - chunk[:, None, :]
        c = triangles[None, :, 2, :] - chunk[:, None, :]
        la = np.linalg.norm(a, axis=2)
        lb = np.linalg.norm(b, axis=2)
        lc = np.linalg.norm(c, axis=2)
        det = np.einsum("pti,pti->pt", a, np.cross(b, c))
        div = (
            la * lb * lc
            + np.einsum("pti,pti->pt", a, b) * lc
            + np.einsum("pti,pti->pt", b, c) * la
            + np.einsum("pti,pti->pt", c, a) * lb
        )
        output[start : start + 256] = (2.0 * np.arctan2(det, div)).sum(axis=1) / (4.0 * math.pi)
    return output


class BrainAssetBuilder:
    """Transforms FlyWire meshes + annotations into a GLB and a neuron layout JSON."""

    def __init__(self, config: BrainAssetConfig | None = None) -> None:
        self.config = config or BrainAssetConfig()
        self._center = np.zeros(3)
        self._scale = 1.0

    def build(self) -> dict[str, Path]:
        cfg = self.config
        cfg.output_dir.mkdir(parents=True, exist_ok=True)

        brain_mesh = load_legacy_ng_mesh((cfg.raw_dir / cfg.brain_mesh_file).read_bytes())
        neuropil_meshes = self._load_neuropil_meshes(cfg.raw_dir / cfg.neuropil_zip_file)

        bounds = brain_mesh.bounds
        self._center = bounds.mean(axis=0)
        self._scale = SCENE_WIDTH_UNITS / float(bounds[1, 0] - bounds[0, 0])

        glb_path = cfg.output_dir / "fly_brain_meshes.glb"
        self._export_glb(brain_mesh, neuropil_meshes, glb_path)

        artifact = ConnectomeArtifact.load(cfg.connectome_path)
        layout_payload = self._build_neuron_layout(artifact, neuropil_meshes)
        layout_path = cfg.output_dir / "fly_brain_neurons.json"
        layout_path.write_text(json.dumps(layout_payload, separators=(",", ":")), encoding="utf-8")
        return {"glb": glb_path, "layout": layout_path}

    # ------------------------------------------------------------------ meshes
    @staticmethod
    def _load_neuropil_meshes(zip_path: Path) -> dict[str, trimesh.Trimesh]:
        meshes: dict[str, trimesh.Trimesh] = {}
        with zipfile.ZipFile(zip_path) as archive:
            for member in archive.namelist():
                name = member.split("/")[-1]
                if member.startswith("__MACOSX") or not name.endswith(".ply"):
                    continue
                mesh = trimesh.load(io.BytesIO(archive.read(member)), file_type="ply", process=True)
                meshes[name[:-4]] = mesh
        return meshes

    def to_scene(self, points_nm: np.ndarray) -> np.ndarray:
        """FAFB nm -> scene units with point reflection (see module docstring)."""
        return -(np.asarray(points_nm, dtype=np.float64) - self._center) * self._scale

    def _to_scene_mesh(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        scene_vertices = self.to_scene(mesh.vertices).astype(np.float32)
        # Point reflection has det = -1: flip winding to keep outward normals.
        flipped_faces = mesh.faces[:, ::-1].copy()
        output = trimesh.Trimesh(vertices=scene_vertices, faces=flipped_faces, process=False)
        output.fix_normals()
        return output

    def _export_glb(
        self,
        brain_mesh: trimesh.Trimesh,
        neuropil_meshes: dict[str, trimesh.Trimesh],
        output_path: Path,
    ) -> None:
        scene = trimesh.Scene()
        scene.add_geometry(self._to_scene_mesh(brain_mesh), node_name="BRAIN_SHELL", geom_name="BRAIN_SHELL")
        for name, mesh in sorted(neuropil_meshes.items()):
            scene.add_geometry(self._to_scene_mesh(mesh), node_name=name, geom_name=name)
        output_path.write_bytes(scene.export(file_type="glb"))

    # ----------------------------------------------------------------- neurons
    def _build_neuron_layout(
        self,
        artifact: ConnectomeArtifact,
        neuropil_meshes: dict[str, trimesh.Trimesh],
    ) -> dict:
        cfg = self.config
        root_ids = [int(nid) for nid in artifact.neuron_ids]
        num_neurons = len(root_ids)

        annotations = (
            pl.read_csv(cfg.raw_dir / cfg.annotations_file, separator="\t", infer_schema_length=0)
            .with_columns(pl.col("root_id").cast(pl.Int64))
            .filter(pl.col("root_id").is_in(root_ids))
            .unique(subset=["root_id"], keep="first")
        )
        ann_by_id = {int(row["root_id"]): row for row in annotations.iter_rows(named=True)}

        # Neuropil layout: deterministic samples inside each neuron's dominant neuropil mesh.
        neuropil_positions = np.zeros((num_neurons, 3), dtype=np.float64)
        neuron_indices_by_neuropil: dict[str, list[int]] = {}
        for idx, neuropil in enumerate(artifact.dominant_neuropils):
            neuron_indices_by_neuropil.setdefault(neuropil, []).append(idx)

        rng = np.random.default_rng(cfg.seed)
        for neuropil, indices in sorted(neuron_indices_by_neuropil.items()):
            mesh = neuropil_meshes.get(neuropil)
            if mesh is None:
                raise KeyError(f"No mesh for dominant neuropil '{neuropil}'.")
            samples = self._sample_inside(mesh, len(indices), rng)
            neuropil_positions[indices] = samples

        # Soma layout: FlyWire soma coordinates (voxel -> nm). Missing -> neuropil sample.
        soma_positions = neuropil_positions.copy()
        soma_found = np.zeros(num_neurons, dtype=bool)
        cell_types: list[str] = []
        for idx, root_id in enumerate(root_ids):
            row = ann_by_id.get(root_id)
            cell_type = ""
            if row is not None:
                cell_type = row.get("cell_type") or row.get("hemibrain_type") or ""
                try:
                    soma_vox = np.array(
                        [float(row["soma_x"]), float(row["soma_y"]), float(row["soma_z"])]
                    )
                    soma_positions[idx] = soma_vox * VOXEL_SIZE_NM
                    soma_found[idx] = True
                except (TypeError, ValueError):
                    pass
            cell_types.append(cell_type)

        edges = self._top_edges(artifact, cfg.top_edge_count)

        neuropil_names = sorted(set(artifact.dominant_neuropils))
        nt_names = sorted(set(artifact.dominant_nts))
        neuropil_index = {name: i for i, name in enumerate(neuropil_names)}
        nt_index = {name: i for i, name in enumerate(nt_names)}

        def flat(points_nm: np.ndarray) -> list[float]:
            return [round(float(v), 4) for v in self.to_scene(points_nm).reshape(-1)]

        return {
            "version": 1,
            "coordinate_frame": "+X fly left, +Y dorsal, +Z anterior; scene units = FAFB nm * scale",
            "scale_units_per_nm": self._scale,
            "center_nm": [float(v) for v in self._center],
            "sources": cfg.sources,
            "num_neurons": num_neurons,
            "cx_count": int(artifact.metadata.cx_count),
            "mb_count": int(artifact.metadata.mb_count),
            "root_ids": [str(rid) for rid in root_ids],
            "cell_types": cell_types,
            "circuit_groups": [0 if g == "CX" else 1 for g in artifact.circuit_groups],
            "neuropil_names": neuropil_names,
            "neuropil_idx": [neuropil_index[n] for n in artifact.dominant_neuropils],
            "nt_names": nt_names,
            "nt_idx": [nt_index[n] for n in artifact.dominant_nts],
            "soma_found": [int(v) for v in soma_found],
            "layout_neuropil": flat(neuropil_positions),
            "layout_soma": flat(soma_positions),
            "edges": edges,
            "cx_neuropils": list(CX_NEUROPILS),
            "mb_neuropils": list(MB_NEUROPILS),
        }

    @staticmethod
    def _sample_inside(mesh: trimesh.Trimesh, count: int, rng: np.random.Generator) -> np.ndarray:
        lower, upper = mesh.bounds
        accepted: list[np.ndarray] = []
        total = 0
        attempts = 0
        while total < count and attempts < 60:
            candidates = rng.uniform(lower, upper, size=(max(count * 4, 256), 3))
            inside = np.abs(generalized_winding_number(candidates, mesh)) > 0.5
            picked = candidates[inside]
            accepted.append(picked)
            total += len(picked)
            attempts += 1
        if total < count:
            raise RuntimeError("Failed to sample enough interior points for neuropil mesh.")
        return np.concatenate(accepted, axis=0)[:count]

    @staticmethod
    def _top_edges(artifact: ConnectomeArtifact, top_k: int) -> dict[str, list]:
        """Strongest directed edges by log1p(syn_count)-derived weight. W[post, pre]."""
        exc_idx = artifact.w_exc_indices.cpu().numpy()
        inh_idx = artifact.w_inh_indices.cpu().numpy()
        exc_val = artifact.w_exc_values.cpu().to(torch.float32).numpy()
        inh_val = artifact.w_inh_values.cpu().to(torch.float32).numpy()

        post = np.concatenate([exc_idx[0], inh_idx[0]])
        pre = np.concatenate([exc_idx[1], inh_idx[1]])
        weight = np.concatenate([exc_val, inh_val])
        polarity = np.concatenate([np.ones_like(exc_val), -np.ones_like(inh_val)])

        keep = pre != post
        post, pre, weight, polarity = post[keep], pre[keep], weight[keep], polarity[keep]
        order = np.argsort(-weight, kind="stable")[:top_k]
        max_weight = float(weight[order[0]]) if len(order) else 1.0
        return {
            "ranking": "top-k by |W| (log1p synapse count, spectrally scaled)",
            "pre": [int(v) for v in pre[order]],
            "post": [int(v) for v in post[order]],
            "polarity": [int(v) for v in polarity[order]],
            "weight_norm": [round(float(v) / max_weight, 4) for v in weight[order]],
        }
