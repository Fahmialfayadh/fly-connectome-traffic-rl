# Can a Fly Control Traffic? (Dapatkah Lalat Mengatur Lalu Lintas?)

Pengujian kelayakan sub-konektom otak lalat buah (*Drosophila melanogaster*, dataset `FlyWire FAFB v783`) sebagai inti rekuren pada kebijakan kendali lampu lalu lintas persimpangan empat arah berbasis *Reinforcement Learning* (PPO). Proyek ini memadukan **2.500 neuron** dari sirkuit **Central Complex (`PB`, `FB`, `EB`, `NO`)** dan **Mushroom Body (`MB_*`)** dengan bobot graf terbeku, berjalan 100% di CPU.

---

## Ringkasan Alur & Artefak Repositori

Proyek ini terdiri atas 5 fase yang dapat dijalankan interaktif melalui Web Lab maupun skrip CLI:

| Fase | Tujuan | Skrip / Antarmuka | Output Artefak |
|---|---|---|---|
| **Fase 1: Ekstraksi Graf** | Saring `1.800` neuron CX + `700` neuron MB dari `connections_princeton.csv`, pisahkan $W_{\text{exc}}$ dan $W_{\text{inh}}$, skalakan *spectral radius* ke `0.95`, serta bangun *null model* acak terkontrol (*global random rewiring*). | `scripts/extract_connectome.py`, `connectome/shuffled.py` | `artifacts/fly_subconnectome.pt`, `artifacts/fly_subconnectome_shuffled.pt` |
| **Fase 2 & 3: Training PPO** | Latih kebijakan `FlyConnectomePolicy` (`fly`), `Shuffled Connectome PPO` (`fly_shuffled`), `MLPBaselinePolicy`, dan `GRUBaselinePolicy` pada 8 environment CPU paralel menggunakan PPO + *decision-step masking* dengan **Live Policy Preview**. | **Tab Training Studio** di Web UI, atau `scripts/train_agents.py` | `artifacts/checkpoints/{fly,fly_shuffled,gru,mlp}_ppo.pt` |
| **Fase 4: Benchmark** | Evaluasi komparatif seluruh model (`Fixed Timer`, `PPO-MLP`, `PPO-GRU`, `Fly Connectome PPO`, `Shuffled Connectome PPO`) di skenario `normal`, `asymmetric`, `rush_hour`, dan `stress_test`, serta uji farmakologi & lesi. | **Tab Benchmark & Connectome** di Web UI, atau `scripts/run_benchmark.py` | `artifacts/benchmarks/benchmark_summary.json`, `benchmark_table.md`, `*.png` |
| **Fase 5: Lab Simulasi, 3D & Dokumentasi** | Jalankan **Inference** secara *real-time* (**3D View** Three.js + sakelar **2D Map** + visualisasi **3D Brain** FlyWire + telemetri adaptif non-fly), eksplorasi arsitektur di **Tab 3D Architecture**, serta laporan teknis di **Tab Documentation**. | `scripts/launch_web_lab.py` (Utama) & `simulation3d/serve.py` (3D Mandiri) | Web UI di `http://127.0.0.1:8080` (`/simulation`, `/training`, `/benchmark`, `/architecture`, `/docs`) |

> **Catatan Penting (Artefak Siap Pakai & Paradigma *Biological Reservoir*):**  
> Repositori ini sudah menyertakan graf hasil ekstraksi (`artifacts/fly_subconnectome.pt`, `artifacts/fly_subconnectome_shuffled.pt`), **4 checkpoint model final Env v2 (~5 juta langkah latih)** di `artifacts/checkpoints/{fly,fly_shuffled,gru,mlp}_ppo.pt`, serta arsip evaluasi di `artifacts/benchmarks/`. Anda dapat langsung menjalankan Web UI atau evaluasi tanpa perlu mengunduh maupun mengekstrak ulang dataset mentah `dataset/` (`>333 MiB`).  
> Matriks konektom FAFB v783 (`2.500` neuron, `87.171` sisi) dibekukan (`requires_grad=False`), sedangkan PPO melatih `70.006` parameter adapter (`input_projection`, `tau_logit`, `actor_head`, `critic_head`). Penjelasan rinci alur 1 episode tersedia di [Panduan Training & Siklus Episode](docs/TRAINING_AND_EPISODE_GUIDE.md).

---

## Cara Menjalankan

Siapkan environment Python (`>= 3.11`, disarankan menggunakan `uv`) dengan paket `numpy`, `polars`, `scipy`, `torch`, `gymnasium`, `aiohttp`, `matplotlib`, `trimesh`, dan `pytest`, lalu jalankan dari root repositori (`PYTHONPATH=src` atau instal editable via `uv pip install -e .`):

### 1. Menjalankan Web Frontend & Backend Server (Direkomendasikan)

```bash
python scripts/launch_web_lab.py --host 127.0.0.1 --port 8080
```

Atau jalankan server 3D Simpang Terpisah (`simulation3d`):

```bash
python simulation3d/serve.py --host 127.0.0.1 --port 8888
```

Lalu buka `http://127.0.0.1:8080` di browser untuk mengakses:
- **Tab Live Simulation**: Simulasi **3D View** Three.js (preset kamera `Top View`, `3D Angle`, `Street`, **`Cinematic`**, `Fullscreen`, navigasi `OrbitControls`, **kompas mata angin hitam-putih** di tengah simpang, HUD sinyal & timer per arah, koreografi lengkung belok dan tabrakan kinematik) serta sakelar **2D Map** dengan **Map Zoom** (`45%`–`125%`), **penambahan kendaraan manual** (klik lajur langsung hingga batas `80` kendaraan/lajur / `640` total), pemilih **Active Model / Controller**, kontrol **Safety Buffer** (`All-Red` vs `Yellow Only`), indikator **Collisions**, probabilitas aksi 4 fase, visualisasi otak 3D (`CX` dan `MB`), *slider* pengali $g_{\text{exc}}$ / $g_{\text{inh}}$, dan tombol lesi sirkuit.
- **Tab Training Studio (`/training`)**: Pemilihan arsitektur kebijakan (`fly`, `fly_shuffled` *Null Model*, `mlp`, `gru`), fitur **Resume from Checkpoint**, pengaturan *timesteps*, panel **Custom PPO Hyperparameters**, opsi penyimpanan checkpoint, grafik optimasi *real-time*, inventaris *checkpoint*, serta **Live Policy Preview** via `/ws/training`.
- **Tab Benchmark & Connectome (`/benchmark`)**: Pemilih model interaktif untuk evaluasi komparatif via `POST /api/benchmark/run`, ringkasan sub-graf `FAFB v783`, grafik perbandingan multi-skenario (`normal`, `asymmetric`, `rush_hour`, `stress_test`), tabel hasil farmakologi/lesi, dan kamus variabel.
- **Tab 3D Architecture (`/architecture`)**: Visualisasi 3D interaktif *end-to-end* untuk `FlyConnectomePolicy` (`x_t -> input_projection -> ConnectomeRecurrentCell -> actor_head / critic_head -> environment`) beserta kontrol **`Cinematic`** dan **`Fullscreen`**, sakelar aliran **Forward Pass** vs **Backward Gradient**, filter status (*Trainable*, *Frozen Bio*, *Runtime*), peta komponen horizontal, dan inspektor detail per modul.
- **Tab Documentation (`/docs`)**: Laporan teknis berbahasa Indonesia dengan daftar isi *sticky*, persamaan matematis, grafik Canvas interaktif tersinkronisasi dari `benchmark_summary.json`, kurva konvergensi training, perbandingan lintas skema, daftar referensi dataset, serta glosari istilah.

### 2. Perintah CLI Opsional (Ekstraksi, Training, Benchmark, Tes)

```bash
# Fase 1 (opsional, jika menyediakan dataset/connections_princeton.csv): Ekstrak ulang sub-connectome (hapus komen jika butuh)
# python scripts/extract_connectome.py

# Fase 2-3: Latih agen dari CLI (ganti parameter sesuai kebutuhan)
python scripts/train_agents.py --agents fly fly_shuffled mlp gru --total-timesteps 163840 --parallel-models 1 --suffix "from_cli"

# Alternatif: Training via Jupyter Notebook untuk eksplorasi interaktif di

"scripts/train_agents.ipynb"

# Resume training
python scripts/train_agents.py --agents fly --total-timesteps 327680 --resume

# Fase 4: Evaluasi komparatif 10 seed per skenario (normal, asymmetric, rush_hour, stress_test)
python scripts/run_benchmark.py

# Jalankan seluruh unit & integration tests
python -m pytest tests/ -q
```

---

## Struktur Direktori Kode

```text
.
├── dataset/                        # Data mentah FlyWire FAFB v783 (CSV)
├── artifacts/                      # Artefak final (sub-connectome, 4 model PPO final, & benchmarks)
│   ├── fly_subconnectome.pt
│   ├── fly_subconnectome_shuffled.pt
│   ├── checkpoints/                # fly_ppo.pt, fly_shuffled_ppo.pt, gru_ppo.pt, mlp_ppo.pt (~5M Env v2)
│   └── benchmarks/                 # JSON, tabel Markdown, & figur hasil evaluasi
├── docs/
│   ├── MASTER_DOC.md                 # Laporan teknis (konektom, null model, lingkungan, PPO, hasil)
│   ├── TRAINING_AND_EPISODE_GUIDE.md # Panduan alur 1 episode (500 langkah), diagram PPO, & jejak episode
│   └── VARIABLES_AND_MATH.md         # Kamus variabel kode, kolisi, dan pemetaan rumus matematika
├── frontend/                       # Antarmuka Web (Modular Partials + CSS + ES6 + Canvas 2D + Three.js)
│   ├── index.html
│   ├── partials/                   # Potongan HTML per tab (simulation, training, benchmark, architecture, documentation)
│   ├── css/
│   │   ├── tokens.css
│   │   ├── layout.css
│   │   └── code.css
│   └── js/
│       ├── main.js
│       ├── intersection_renderer.js
│       ├── intersection3d_renderer.js
│       ├── sim3d/                  # Modul 3D Simpang Empat terintegrasi (world, signals, vehicles)
│       ├── connectome_renderer.js
│       ├── model_telemetry_renderer.js
│       ├── brain3d_renderer.js
│       ├── architecture_view.js
│       ├── training_studio.js
│       ├── benchmark_view.js
│       └── documentation_view.js
├── simulation3d/                   # Aplikasi & server mandiri 3D Persimpangan Three.js (port 8888)
│   ├── index.html
│   ├── serve.py
│   ├── css/
│   └── js/
├── scripts/                        # CLI entrypoints
│   ├── extract_connectome.py
│   ├── train_agents.py
│   ├── train_agents.ipynb          # Jupyter Notebook alternatif untuk training agen PPO
│   ├── run_benchmark.py
│   ├── launch_web_lab.py
│   └── launch_sandbox.py           # Utilitas headless Pygame opsional
├── src/fly_traffic/                # Paket backend modular OOP
│   ├── __init__.py
│   ├── config.py
│   ├── connectome/                 # Ekstraksi, shuffled null model, aset 3D, & serialisasi sparse CSR
│   ├── environment/                # TrafficIntersectionEnv (Gymnasium) & profil kedatangan
│   ├── agents/                     # ConnectomeRecurrentCell, Fly/MLP/GRU policies, FixedTimer
│   ├── training/                   # RecurrentRolloutBuffer & PPOTrainer
│   ├── evaluation/                 # AgentEvaluator & BenchmarkPlotter
│   ├── web/                        # Server aiohttp, SimulationService, TrainingJobManager
│   └── ui/                         # Renderer Pygame opsional
├── tests/                          # Pengujian unit & integrasi (pytest)
├── pyproject.toml
├── uv.lock
└── README.md
```

---

## Fisika Kolisi & Live Policy Preview

- **Pemodelan Kolisi (`TrafficIntersectionEnv._evaluate_collisions`)**:
  - **`red_light_conflict`**: Lajur merah yang menunggu melebihi ambang `red_light_wait_threshold_s = 150.0` s berpeluang menerobos lampu merah (hingga `red_light_max_prob = 0.14`) dan bertabrakan dengan lajur hijau yang sedang melaju.
  - **`box_clearance_conflict`**: Saat terjadi perpindahan fase, lajur pelepas sebelumnya berpeluang `dilemma_zone_prob = 0.18` menyisakan kendaraan *straggler* berumur 2 langkah di kotak persimpangan (`_box_stragglers`). Dengan *safety buffer* bawaan (`yellow_steps = 2`, `all_red_steps = 1`), straggler terselesaikan sebelum fase baru menjadi `GREEN`; konflik muncul apabila total jeda antar-hijau diturunkan di bawah 2 langkah (`yellow_steps <= 1` dan `all_red_steps = 0`).
  - Setiap kolisi mengurangi *reward* langkah sebesar `collision_penalty * reward_scale` (`80.0 * 0.005 = 0.400`) dan dirender sebagai indikator bahaya pada 3D View maupun Canvas 2D.
- **Kapasitas Live Simulation & Normalisasi Observasi (`SimulationService` / `TrafficIntersectionEnv`)**:
  - Di tab **Live Simulation**, batas kapasitas per lajur diatur ke `hard_max_lane_capacity = 80` kendaraan/lajur (`640` kendaraan total maksimal) untuk menampung penambahan manual dan mode zoom-out `45%`–`125%`, sementara normalisasi observasi antrean `obs[0:8]` tetap menggunakan `queue_obs_norm_capacity = 30.0` dan dijepit ke `[0.0, 2.0]` agar kompatibel dengan model PPO yang dilatih pada kapasitas latihan standar (`30`/lajur).
- **Live Policy Preview (`TrainingJobManager._rollout_preview_frames`)**:
  - Selama training PPO berjalan *headless* di CPU pada 8 environment vektor (`SyncVectorEnv`), `TrainingJobManager` mengeksekusi 4 tick deterministik pada satu environment `rush_hour` terpisah setiap selesai 1 update PPO dan mengirim `preview_frames` lewat WebSocket `/ws/training`. Tab **Training Studio** memutar frame tersebut secara berkala tanpa memperlambat proses optimasi CPU.

---

## Ringkasan Perubahan *Shuffled Connectome* (*Null Model*)

Untuk mengisolasi pengaruh **topologi biologis FAFB v783**, modul [`src/fly_traffic/connectome/shuffled.py`](src/fly_traffic/connectome/shuffled.py) (`create_shuffled_connectome_artifact()` & `load_or_create_shuffled_connectome()`) membangun graf kontrol acak terkontrol (*global random edge rewiring*, `seed=42`) dengan menempatkan ulang seluruh pasangan `(post, pre)` secara seragam pada matriks `2.500 × 2.500`:

| Properti | Fly Connectome Asli (`fly_subconnectome.pt`) | Shuffled Connectome (`fly_subconnectome_shuffled.pt`) | Keterangan Perubahan |
|---|---|---|---|
| **Jumlah Neuron (`CX + MB`)** | `2.500` (`1.800` + `700`) | `2.500` (`1.800` + `700`) | Sama persis (metadata wilayah & label dipertahankan) |
| **Sisi Eksitatori (`W_exc`)** | `36.908` sisi | `36.908` sisi | Jumlah sisi & himpunan bobot relatif sama, pasangan `(post, pre)` diacak |
| **Sisi Inhibitori (`W_inh`)** | `50.263` sisi | `50.263` sisi | Jumlah sisi & himpunan bobot relatif sama, pasangan `(post, pre)` diacak |
| **Kejarangan (*Sparsity*)** | `98,6053%` | `98,6053%` | Identik (`87.171` sisi non-nol dari `6.250.000` sel) |
| **Koneksi Lintas Wilayah (`CX <-> MB`)** | `2.880` sisi | `35.430` sisi | Modularitas biologis hancur; koneksi tersebar acak antar kluster |
| ***Raw Spectral Radius*** (`W_exc - W_inh`) | `113,1494` | `0,1302` | Struktur spektral sebelum normalisasi berubah |
| ***Scaled Spectral Radius*** | `0,9500` | `0,9500` | Sama-sama dinormalisasi ulang ke `0,95` |
| **Parameter Adapter PPO** | `70.006` | `70.006` | Menggunakan arsitektur `FlyConnectomePolicy` yang sama |

---

## Hasil Benchmark Terbaru (10 Seed per Skenario, 500 Langkah / Episode)

Tabel di bawah disinkronkan langsung dari `artifacts/benchmarks/benchmark_table.md` (`enable_collisions=True`, `min_green_steps=6`, `yellow_steps=2`, `all_red_steps=1`, `red_light_wait_threshold_s=150.0`). Keempat *checkpoint* final Env v2 tersebut telah dipasang sebagai model bawaan `artifacts/checkpoints/{fly,fly_shuffled,gru,mlp}_ppo.pt`:

| Skenario | Agen | Parameter Terlatih | Mean Reward | Mean Queue | Mean Wait (s) | Mean Throughput | Mean Switches | Mean Collisions | Latency (ms/step) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `normal` | Fly Connectome PPO (`fly_ppo_new_env`) | 70.006 | -35.446 | 9.32 | 25.82 | 352.4 | 54.7 | 0.00 | 0.265 |
| `normal` | Shuffled Connectome PPO (`fly_shuffled_ppo_new_env`) | 70.006 | -35.630 | 9.37 | 25.99 | 353.2 | 55.0 | 0.00 | 0.289 |
| `normal` | PPO-GRU (`gru_ppo_new_env`) | 70.041 | -34.923 | 9.32 | 26.07 | 348.0 | 53.5 | 0.00 | 0.176 |
| `normal` | PPO-MLP (`mlp_ppo_new_env`) | 69.506 | -34.605 | 9.21 | 25.75 | 349.1 | 52.9 | 0.00 | 0.128 |
| `normal` | Fixed Timer (`10-6-10-6`) | - | -40.626 | 11.01 | 30.69 | 348.8 | 45.0 | 0.00 | 0.002 |
| `asymmetric` | Fly Connectome PPO (`fly_ppo_new_env`) | 70.006 | -48.370 | 13.75 | 23.57 | 571.5 | 55.0 | 0.00 | 0.282 |
| `asymmetric` | Shuffled Connectome PPO (`fly_shuffled_ppo_new_env`) | 70.006 | -47.624 | 13.88 | 23.83 | 570.1 | 55.0 | 0.00 | 0.264 |
| `asymmetric` | PPO-GRU (`gru_ppo_new_env`) | 70.041 | -47.893 | 13.34 | 22.63 | 577.1 | 53.6 | 0.00 | 0.132 |
| `asymmetric` | PPO-MLP (`mlp_ppo_new_env`) | 69.506 | -47.616 | 13.47 | 23.27 | 565.5 | 53.5 | 0.00 | 0.089 |
| `asymmetric` | Fixed Timer (`10-6-10-6`) | - | -61.537 | 19.12 | 32.92 | 569.5 | 45.0 | 0.00 | 0.004 |
| `rush_hour` | Fly Connectome PPO (`fly_ppo_new_env`) | 70.006 | -79.574 | 25.65 | 27.96 | 886.7 | 54.8 | 0.00 | 0.393 |
| `rush_hour` | Shuffled Connectome PPO (`fly_shuffled_ppo_new_env`) | 70.006 | -79.431 | 25.60 | 28.05 | 879.7 | 54.3 | 0.00 | 0.262 |
| `rush_hour` | PPO-GRU (`gru_ppo_new_env`) | 70.041 | -80.937 | 26.01 | 28.09 | 894.5 | 53.9 | 0.00 | 0.098 |
| `rush_hour` | PPO-MLP (`mlp_ppo_new_env`) | 69.506 | -76.871 | 24.71 | 27.18 | 879.3 | 54.0 | 0.00 | 0.079 |
| `rush_hour` | Fixed Timer (`10-6-10-6`) | - | -104.742 | 35.43 | 38.61 | 855.9 | 45.0 | 0.00 | 0.005 |
| `stress_test` | Fly Connectome PPO (`fly_ppo_new_env`) | 70.006 | -399.038 | 144.89 | 88.38 | 1448.8 | 54.9 | 5.00 | 0.266 |
| `stress_test` | Shuffled Connectome PPO (`fly_shuffled_ppo_new_env`) | 70.006 | -395.579 | 140.52 | 84.97 | 1460.1 | 54.9 | 14.00 | 0.264 |
| `stress_test` | PPO-GRU (`gru_ppo_new_env`) | 70.041 | -361.209 | 129.12 | 78.82 | 1468.7 | 52.6 | 8.60 | 0.099 |
| `stress_test` | PPO-MLP (`mlp_ppo_new_env`) | 69.506 | -381.887 | 131.83 | 78.69 | 1487.1 | 53.8 | 25.20 | 0.085 |
| `stress_test` | Fixed Timer (`10-6-10-6`) | - | -365.067 | 126.19 | 76.47 | 1482.8 | 45.0 | 20.90 | 0.002 |

Figur hasil eksperimen tersedia di `artifacts/benchmarks/`: `scenario_comparison.png`, `training_curves.png`, `pharmacology_ablation.png`, dan `experiment_evolution_comparison.png`.

---

## Dokumentasi Teknis

- [Laporan Teknis Utama](docs/MASTER_DOC.md): Pembahasan menyeluruh konstruksi sub-konektom FAFB v783, *Shuffled Null Model*, lingkungan lalu lintas, paradigma reservoir biologis & adapter, PPO, serta hasil evaluasi.
- [Panduan Training & Siklus Episode](docs/TRAINING_AND_EPISODE_GUIDE.md): Diagram Mermaid siklus 1 episode (`500` langkah), alur loop PPO CPU + Live Policy Preview, serta referensi jejak episode historis.
- [Kamus Variabel & Matematika](docs/VARIABLES_AND_MATH.md): Pemetaan simbol persamaan biologis, observasi, reward, kolisi, *safety buffer*, dan hyperparameter PPO ke nama variabel kode.
