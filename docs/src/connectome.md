# Konstruksi & Manipulasi Sub-Konektom (`src/fly_traffic/connectome/`)

## 1. Alur Ekstraksi Graf (`ConnectomePreprocessor.extract`)

Pipeline membaca `connections_princeton.csv` secara *lazy* via Polars (`pl.scan_csv`) lalu membangun dua matriks jarang (*sparse*) eksitatori dan inhibitori yang dinormalisasi ke *spectral radius* target (`0.95`).

1. **Seleksi Neuron (`_select_top_neurons`)**: menghitung total `syn_count` insiden per `root_id` pada neuropil target, mengambil `1.800` neuron teratas dari **CX** (`PB`, `FB`, `EB`, `NO`), lalu `700` neuron teratas dari **MB** (`MB_CA_L/R`, `MB_ML_L/R`, `MB_PED_L/R`, `MB_VL_L/R`) dengan mengecualikan neuron yang sudah terpilih di CX.
2. **Induksi Sub-graf & Pemisahan Polaritas**: menyaring seluruh sinaps dengan `pre_root_id` dan `post_root_id` berada dalam `selected_root_ids` (`2.500` neuron) sehingga jembatan `CX <-> MB` (`2.880` sisi) ikut terbawa, lalu memisahkan `nt_type` menjadi `W_exc` (`ACH`) dan `W_inh` (`GABA`, `GLUT`).
3. **Pembobotan Sinaptik & Spectral Scaling**:

```python
raw_net_matrix = (raw_excitatory_matrix - raw_inhibitory_matrix).astype(np.float64)
raw_spectral_radius = self._compute_spectral_radius(raw_net_matrix)
scaling_factor = (
    self.config.target_spectral_radius / raw_spectral_radius
    if raw_spectral_radius > 0.0
    else 1.0
)
scaled_excitatory_matrix = (raw_excitatory_matrix * scaling_factor).astype(np.float32)
scaled_inhibitory_matrix = (raw_inhibitory_matrix * scaling_factor).astype(np.float32)
```

- Setiap elemen matriks dikompresi dengan $\log(1 + \text{syn\_count}_{ij})$ sebelum normalisasi spektral (`113.1494 -> 0.9500`).

---

## 2. Skema `ConnectomeArtifact` & `ConnectomeMetadata`

| Field | Tipe | Bentuk / Isi | Keterangan |
|---|---|---|---|
| `w_exc_indices` | `torch.Tensor` (`int64`) | `[2, E_exc]` (`36.908` sisi) | Koordinat COO `[post_idx, pre_idx]` koneksi eksitatori. |
| `w_exc_values` | `torch.Tensor` (`float32`) | `[E_exc]` | Bobot eksitatori hasil `log1p` dan scaling spektral. |
| `w_inh_indices` | `torch.Tensor` (`int64`) | `[2, E_inh]` (`50.263` sisi) | Koordinat COO `[post_idx, pre_idx]` koneksi inhibitori. |
| `w_inh_values` | `torch.Tensor` (`float32`) | `[E_inh]` | Bobot inhibitori (disimpan positif, disubtraksi saat propagasi). |
| `neuron_ids` | `list[int]` | Panjang `2.500` | FlyWire v783 `root_id` terurut. |
| `circuit_groups` | `list[str]` | Panjang `2.500` | Label wilayah (`"CX"` atau `"MB"`). |
| `dominant_neuropils` | `list[str]` | Panjang `2.500` | Neuropil asal terbanyak per neuron (dipakai untuk layout 3D & lesi). |
| `dominant_nts` | `list[str]` | Panjang `2.500` | Neurotransmiter dominan (`ACH`, `GABA`, `GLUT`). |
| `metadata` | `ConnectomeMetadata` | Dataclass | Menyimpan `num_neurons`, `cx_count`, `mb_count`, `exc_edge_count`, `inh_edge_count`, `cross_region_edges`, `sparsity`, `raw_spectral_radius`, `scaled_spectral_radius`. |

- `artifact.to_sparse_csr_tensors(device="cpu")` mengonversi kedua tensor COO ke format CSR *coalesced* untuk perkalian matriks cepat di CPU (`torch.sparse.mm`).
- Serialisasi `save()` / `load()` hanya menyimpan tensor numerik dan tipe primitif Python agar aman dimuat dengan `torch.load(..., weights_only=True)`.

---

## 3. *Shuffled Connectome Null Model* (`create_shuffled_connectome_artifact`)

Fungsi `_rewire_edges_globally()` menyatukan pasangan duplikat terlebih dahulu, lalu menarik ulang posisi `(row, col)` tanpa pengembalian dari `num_neurons * num_neurons` slot secara seragam:

```python
flat_indices = rng.choice(total_slots, size=edge_count, replace=False)
rows = (flat_indices // num_neurons).astype(np.int64)
cols = (flat_indices % num_neurons).astype(np.int64)
weights = coalesced.data.astype(np.float32, copy=True)
```

| Invariant yang Dipertahankan | Nilai | Yang Berubah |
|---|---|---|
| `num_neurons`, `cx_count`, `mb_count` | `2.500` (`1.800` + `700`) | Struktur modular CX/MB teracak seragam |
| `exc_edge_count` / `inh_edge_count` | `36.908` / `50.263` | `cross_region_edges` naik (`2.880 -> 35.430`) |
| `sparsity` | `0.986053` (`98,6053%`) | `raw_spectral_radius` berubah (`113,1494 -> 0,1302`) |
| `scaled_spectral_radius` | `0.9500` | Diskalakan ulang setelah *rewiring* |

---

## 4. Konversi Aset Anatomi 3D (`BrainAssetBuilder`)

Memetakan koordinat voxel FAFB (`VOXEL_SIZE_NM = [4.0, 4.0, 40.0]` nm) ke ruang tiga dimensi scene Three.js (`SCENE_WIDTH_UNITS = 10.0`) dengan pembalikan sumbu karena ruang voxel FAFB cermin-terbalik (`+X = fly left`, `+Y = dorsal`, `+Z = anterior`). `BrainAssetBuilder.build()` menghasilkan `fly_brain_meshes.glb`, `fly_brain_neurons.json`, dan `top_edge_count = 3.000` sisi terkuat untuk pulsa arah sinapsis.
