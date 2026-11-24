# Referensi Skrip CLI & Notebook (`docs/scripts/cli.md`)

Seluruh skrip dijalankan dari root repositori (`uv run python ...`, `PYTHONPATH=src`, atau setelah instalasi editable `uv pip install -e .`).

---

## 1. `scripts/extract_connectome.py`
Menjalankan `ConnectomePreprocessor.extract()` untuk menyaring `1.800` CX + `700` MB neuron dari CSV FAFB v783 dan menyimpan artefak `.pt`.

| Argumen | Tipe | Default | Keterangan |
|---|---|---|---|
| `--dataset` | `Path` | `ConnectomeConfig.dataset_path` | Path ke `connections_princeton.csv`. |
| `--output` | `Path` | `ConnectomeConfig.output_path` | Path file artefak `.pt` keluaran. |
| `--target-radius` | `float` | `ConnectomeConfig.target_spectral_radius` (`0.95`) | Target *spectral radius* matriks bersih $(W_{\text{exc}} - W_{\text{inh}})$. |

---

## 2. `scripts/build_brain_assets.py`
Memanggil `BrainAssetBuilder().build()` tanpa argumen CLI; membaca `dataset/brain3d/{brain_mesh_v3.bin, JFRC2NP.surf.fw.zip, neuron_annotations.tsv}` dan `artifacts/fly_subconnectome.pt`, lalu menulis `fly_brain_meshes.glb` dan `fly_brain_neurons.json` ke `frontend/assets/brain/`.

---

## 3. `scripts/train_agents.py`
Melatih satu atau beberapa kebijakan PPO (`fly`, `fly_shuffled`, `mlp`, `gru`) secara sekuensial atau multi-proses CPU (`spawn`).

| Argumen | Tipe | Default | Keterangan |
|---|---|---|---|
| `--agents` | `list[str]` | `["fly", "mlp", "gru"]` | Pilihan: `fly`, `fly_shuffled`, `mlp`, `gru`. |
| `--total-timesteps` | `int` | `PPOConfig.total_timesteps` (`163840`) | Target langkah latih per agen (skema final dilaporkan memakai `5_000_000`). |
| `--num-envs` | `int` | `PPOConfig.num_envs` (`8`) | Jumlah `SyncVectorEnv` paralel per agen. |
| `--seed` | `int` | `PPOConfig.seed` (`42`) | Seed NumPy dan PyTorch. |
| `--connectome` | `Path` | `ConnectomeConfig.output_path` | Path ke `fly_subconnectome.pt` (`fly_shuffled` otomatis memuat/membuat file `*_shuffled.pt` di sebelahnya). |
| `--checkpoints-dir` | `Path` | `PPOConfig.checkpoints_dir` | Direktori penyimpanan checkpoint. |
| `--suffix` | `str` | `""` | Akhiran nama berkas (`{model_type}_ppo{suffix}.pt`). |
| `--parallel-models` | `int` | `1` | Jumlah agen yang dilatih bersamaan via `ProcessPoolExecutor` (`1`, `2`, atau `4`). Di mode paralel, tiap worker mengunci `torch.set_num_threads(1)`. |
| `--resume` | `flag` | `False` | Melanjutkan optimasi dari checkpoint yang cocok di `checkpoints-dir` bila sudah ada. |
| `--lr` | `float` | `PPOConfig.learning_rate` (`8e-4`) | Laju belajar awal optimizer Adam. |
| `--copy-checkpoints` | `flag` | `False` | Menyalin checkpoint `{model}_ppo{suffix}.pt` ke `checkpoints_dir/backups/` dengan timestamp (mewajibkan `--suffix`). |

---

## 4. `scripts/run_benchmark.py`
Mengevaluasi kelima pengendali (`Fly Connectome PPO`, `Shuffled Connectome PPO`, `PPO-GRU`, `PPO-MLP`, dan `Fixed Timer`) melalui `AgentEvaluator` pada 4 skenario (`10` seed) + uji farmakologi/lesi, lalu menulis JSON, tabel Markdown, serta figur PNG.

| Argumen | Tipe | Default | Keterangan |
|---|---|---|---|
| `--checkpoints-dir` | `Path` | `PPOConfig.checkpoints_dir` | Direktori berisi `*_ppo.pt`. |
| `--connectome` | `Path` | `ConnectomeConfig.output_path` | Path ke `fly_subconnectome.pt` (`fly_shuffled` otomatis memuat/membuat `*_shuffled.pt`). |
| `--output-dir` | `Path` | `BenchmarkConfig.output_dir` | Direktori keluaran `benchmark_summary.json`, `benchmark_table.md`, `training_curves.png`, `scenario_comparison.png`, `mean_reward.png`, `mean_collisions.png`, dan `pharmacology_ablation.png`. |

---

## 5. `scripts/launch_web_lab.py`
Menjalankan `TrafficLabServer` berbasis `aiohttp`.

| Argumen | Tipe | Default | Keterangan |
|---|---|---|---|
| `--host` | `str` | `"127.0.0.1"` | Antarmuka host bind server HTTP/WS. |
| `--port` | `int` | `8080` | Port TCP server Web Lab. |
| `--connectome` | `Path` | `ConnectomeConfig.output_path` | Path ke `fly_subconnectome.pt`. |
| `--checkpoints-dir` | `Path` | `PPOConfig.checkpoints_dir` | Direktori inventaris checkpoint `.pt`. |
| `--benchmarks-dir` | `Path` | `DEFAULT_ARTIFACTS_DIR / "benchmarks"` | Direktori hasil benchmark & gambar PNG yang disajikan pada `/artifacts/benchmarks/`. |

---

## 6. `simulation3d/serve.py`
Menjalankan penampil 3D simpang empat mandiri (`Traffic3DServer`) yang berbagi modul Three.js dari `frontend/js/sim3d/`.

| Argumen | Tipe | Default | Keterangan |
|---|---|---|---|
| `--host` | `str` | `"127.0.0.1"` | Antarmuka host bind server HTTP/WS. |
| `--port` | `int` | `8888` | Port TCP server 3D mandiri (juga mendukung argumen posisional opsional). |

---

## 7. `scripts/launch_sandbox.py`
Menjalankan aplikasi desktop Pygame `TrafficConnectomeSandboxApp` (memerlukan paket opsional `pygame` dan `pillow`).

| Argumen | Tipe | Default | Keterangan |
|---|---|---|---|
| `--connectome` | `Path` | `ConnectomeConfig.output_path` | Path ke `fly_subconnectome.pt`. |
| `--checkpoints-dir` | `Path` | `PPOConfig.checkpoints_dir` | Direktori checkpoint terlatih. |
| `--headless-screenshot` | `Path \| None` | `None` | Jika diisi, render simulasi tanpa jendela display lalu simpan PNG ke path ini. |
| `--steps` | `int` | `45` | Jumlah langkah simulasi sebelum screenshot headless diambil. |

---

## 8. `notebooks/train_agents.ipynb`
Notebook Jupyter interaktif untuk eksperimen pelatihan PPO dan evaluasi terisolasi (`artifacts/benchmarks/notebook_*/`), kompatibel untuk eksekusi lokal maupun remote (mis. Google Colab).
