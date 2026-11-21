# Drosophila Connectome RL Traffic Controller

Sistem kendali lampu lalu lintas berbasis *Reinforcement Learning* (PPO) yang menggantikan lapisan tersembunyi MLP/GRU standar dengan sub-graf biologis otak lalat buah (*Drosophila melanogaster*, dataset `FAFB v783`) berisi **2.500 neuron** dari sirkuit **Central Complex (`PB`, `FB`, `EB`, `NO`)** dan **Mushroom Body (`MB_*`)**, berjalan 100% pada CPU.

---

## Ringkasan Alur & Status Workspace

Proyek ini terdiri atas 5 fase yang sudah dapat dijalankan secara terpisah maupun interaktif lewat antarmuka web:

| Fase | Tujuan | Skrip / Antarmuka | Output Artefak |
|---|---|---|---|
| **Fase 1: Ekstraksi Graf** | Saring 1.800 neuron CX + 700 neuron MB dari `connections_princeton.csv`, pisahkan $W_{\text{exc}}$ dan $W_{\text{inh}}$, skalakan *spectral radius* ke `0.95`, serta bangun *null model* acak terkontrol (*global random rewiring*). | `scripts/extract_connectome.py`, `connectome/shuffled.py` | `artifacts/fly_subconnectome.pt`, `artifacts/fly_subconnectome_shuffled.pt` |
| **Fase 2 & 3: Training PPO** | Latih kebijakan `FlyConnectomePolicy` (`fly`), `Shuffled Connectome PPO` (`fly_shuffled`), `MLPBaselinePolicy`, dan `GRUBaselinePolicy` pada 8 environment CPU paralel menggunakan PPO + *decision-step masking* dan pantau **Live Policy Preview** (4 tick `rush_hour` per update). | **Tab Training Studio** di Web UI, atau `scripts/train_agents.py` | `artifacts/checkpoints/{fly,fly_shuffled,mlp,gru}_ppo.pt` |
| **Fase 4: Benchmark** | Evaluasi komparatif model yang dipilih pengguna (`Fixed Timer`, `PPO-MLP`, `PPO-GRU`, `Fly Connectome PPO`, `Shuffled Connectome PPO`, maupun *checkpoint* kustom `.pt`) di skenario `normal`, `asymmetric`, `rush_hour` (termasuk grafik & metrik `mean_collisions`), serta uji farmakologi & lesi. | **Tab Benchmark & Connectome** di Web UI, atau `scripts/run_benchmark.py` | `artifacts/benchmarks/benchmark_summary.json`, `benchmark_table.md`, `*.png` |
| **Fase 5: Lab Simulasi & Eksplorasi 3D** | Jalankan **Inference** secara *real-time* (60 FPS Canvas 2D + visualisasi 3D otak FlyWire), inspeksi end-to-end arsitektur pada **Tab 3D Architecture** (*Forward Pass* vs *Backward Gradient Flow* + panduan matematis), deteksi kolisi, *Safety Buffer*, dan kontrol $g_{\text{exc}} / g_{\text{inh}}$ + lesi. | `scripts/launch_web_lab.py` (Utama) | Web UI di `http://127.0.0.1:8080` |

> **Catatan Penting (Inference vs Training):**  
> Saat membuka Web UI pertama kali, tab **Live Simulation** berjalan dalam mode **INFERENCE** menggunakan *checkpoint* terlatih yang sudah tersedia di `artifacts/checkpoints/` (hasil training 163.840 langkah sebelumnya).  
> Jika Anda ingin menjalankan **TRAINING** sendiri dari nol dan memantau kurva *reward*, *loss*, *entropy*, serta **Live Policy Preview** secara langsung, buka tab **Training Studio** di Web UI (atau jalankan `scripts/train_agents.py`).  
> Penjelasan alur lengkap 1 episode (300 langkah), diagram Mermaid loop PPO, dan penelusuran episode nyata tersedia di [Panduan Training & Siklus Episode](docs/TRAINING_AND_EPISODE_GUIDE.md).

---

## Cara Menjalankan

Gunakan environment Python pada `/cache/ml/.venv/bin/python`:

### 1. Menjalankan Web Frontend & Backend Server (Direkomendasikan)

```bash
/cache/ml/.venv/bin/python scripts/launch_web_lab.py --host 127.0.0.1 --port 8080
```

Lalu buka `http://127.0.0.1:8080` di browser untuk mengakses:
- **Tab Live Simulation**: Simulasi 2D lampu lalu lintas dengan kontrol **Map Zoom** (`45%`–`125%` lewat tombol zoom maupun roda mouse), **penambahan kendaraan manual** (**klik langsung salah satu dari 8 lajur masuk pada peta** dengan pilihan `+1`/`+5`/`+10`, atau gunakan `Target Lane` + `Add to Lane`, `+5 Random`, serta `+24 Surge (3/lane)` dengan batas aman `80` kendaraan/lajur atau `640` total maksimal), pemilih dinamis **Active Model / Controller** (menampilkan seluruh file `.pt` lokal di `artifacts/checkpoints/` beserta nama file dan jumlah langkah latihnya), kontrol **Safety Buffer** (`Yellow + All-Red` vs `Yellow Only`), indikator visual serta penghitung **Collisions** (`red_light_conflict` & `box_clearance_conflict`), grafik probabilitas keputusan 4 fase, *heatmap* 50x50 neuron (`CX` dan `MB`), *slider* pengali $g_{\text{exc}}$ / $g_{\text{inh}}$, dan tombol lesi sirkuit.
- **Tab Training Studio**: Pemilihan arsitektur kebijakan (`fly`, `fly_shuffled` *Null Model*, `mlp`, `gru`), pengaturan *timesteps* hingga **Mode Maximal** (`524.288` langkah) dan **Custom Timesteps** (`1.024`–`2.000.000`), panel **Custom PPO Hyperparameters** (`gamma`, `gae_lambda`, `clip_coef`, `ent_coef`, `vf_coef`, `update_epochs`, `rollout_steps`, `sequence_chunk_len`, `num_minibatches`, `anneal_lr`, dan opsi **Save-on-Stop** `save_on_cancel`), pilihan **Model File Save Mode** (**Create a new model file (.pt)**, **Overwrite default**, atau **Dry run**), grafik optimasi *real-time* (termasuk `mean_collisions`), inventaris seluruh *checkpoint* lokal dengan tombol **Load in Live Sim**, serta **Live Policy Preview** (`rush_hour`, 4 snapshot per update PPO via `/ws/training`).
- **Tab Benchmark & Connectome**: Pemilih model interaktif (*checkbox* untuk memilih kombinasi model bawaan, `Shuffled Connectome PPO`, maupun *checkpoint* kustom `.pt` yang ingin dibandingkan via `POST /api/benchmark/run`), ringkasan sub-graf `FAFB v783`, grafik perbandingan multi-skenario (`mean_reward`, `mean_wait_time_s`, dan `mean_collisions`), tabel hasil farmakologi/lesi, dan kamus variabel.
- **Tab 3D Architecture**: Visualisasi 3D interaktif *end-to-end* untuk `FlyConnectomePolicy` (`x_t -> input_projection -> ConnectomeRecurrentCell -> actor_head / critic_head -> environment`), sakelar aliran **Forward Pass** vs **Backward Gradient**, filter status (*Trainable*, *Frozen Bio*, *Runtime*), peta komponen horizontal, inspektor detail per modul, serta panduan matematis 4 tahap.

### 2. Perintah CLI Opsional (Ekstraksi, Training, Benchmark, Tes)

```bash
# Fase 1: Ekstrak ulang sub-connectome dari dataset CSV
/cache/ml/.venv/bin/python scripts/extract_connectome.py

# Fase 2-3: Latih semua agen dari CLI (default 163.840 timesteps per agen)
/cache/ml/.venv/bin/python scripts/train_agents.py --agents fly mlp gru --total-timesteps 163840

# Fase 4: Evaluasi komparatif 10 seed per skenario dan generate grafik PNG
/cache/ml/.venv/bin/python scripts/run_benchmark.py

# Jalankan seluruh unit & integration tests
/cache/ml/.venv/bin/python -m pytest tests/ -v
```

---

## Struktur Direktori Kode

```text
.
├── dataset/                        # Data mentah FlyWire FAFB v783 (CSV)
├── artifacts/                      # Artefak tergenerasi (sub-connectome, checkpoints, benchmarks)
├── docs/
│   ├── MASTER_DOC.md                 # Laporan teknis lengkap (konektom, null model, lingkungan, PPO, hasil)
│   ├── TRAINING_AND_EPISODE_GUIDE.md # Panduan alur 1 episode (300 langkah), diagram PPO & penelusuran nyata
│   └── VARIABLES_AND_MATH.md         # Kamus lengkap variabel kode, kolisi, dan pemetaan rumus matematika
├── frontend/                       # Antarmuka Web (HTML + OKLCH CSS + Modular Vanilla ES6 + Canvas 2D + Three.js)
│   ├── index.html
│   ├── css/
│   │   ├── tokens.css
│   │   └── layout.css
│   └── js/
│       ├── main.js
│       ├── intersection_renderer.js
│       ├── connectome_renderer.js
│       ├── brain3d_renderer.js
│       ├── architecture_view.js
│       ├── training_studio.js
│       └── benchmark_view.js
├── scripts/                        # CLI entrypoints
│   ├── extract_connectome.py
│   ├── train_agents.py
│   ├── run_benchmark.py
│   ├── launch_web_lab.py
│   └── launch_sandbox.py           # Utilitas headless Pygame opsional
├── src/fly_traffic/                # Paket backend modular OOP
│   ├── config.py
│   ├── connectome/                 # Ekstraksi & serialisasi sparse CSR
│   ├── environment/                # TrafficIntersectionEnv (Gymnasium) & profil kedatangan
│   ├── agents/                     # ConnectomeRecurrentCell, Fly/MLP/GRU policies, FixedTimer
│   ├── training/                   # RecurrentRolloutBuffer & PPOTrainer
│   ├── evaluation/                 # AgentEvaluator & BenchmarkPlotter
│   ├── web/                        # Server aiohttp, SimulationService, TrainingJobManager
│   └── ui/                         # Renderer Pygame opsional
├── tests/                          # Pengujian unit & integrasi (pytest)
├── PLAN.md
├── PRODUCT.md
├── DESIGN.md
└── README.md
```

---

## Fisika Kolisi & Live Policy Preview

- **Pemodelan Kolisi (`TrafficIntersectionEnv._evaluate_collisions`)**:
  - **`red_light_conflict`**: Lajur merah yang menunggu melebihi ambang `red_light_wait_threshold_s = 36.0` s berpeluang menerobos lampu merah (hingga `red_light_max_prob = 0.14`) dan bertabrakan dengan lajur hijau yang sedang melaju.
  - **`box_clearance_conflict`**: Saat terjadi perpindahan fase, lajur pelepas sebelumnya berpeluang `dilemma_zone_prob = 0.18` menyisakan kendaraan *straggler* di kotak persimpangan (`_box_stragglers`). Dengan *safety buffer* `all_red_steps = 1` (`Yellow + All-Red`), straggler terselesaikan sebelum fase baru menjadi `GREEN`; jika diubah menjadi `all_red_steps = 0` (`Yellow Only` lewat kontrol UI atau `set_safety_buffer(0)`), konflik kotak persimpangan langsung meningkat.
  - Setiap kolisi mengurangi *reward* langkah sebesar `collision_penalty * reward_scale` (`15.0 * 0.005 = 0.075`) dan dirender sebagai indikator bahaya pada Canvas 2D.
- **Kapasitas Live Simulation & Normalisasi Observasi (`SimulationService` / `TrafficIntersectionEnv`)**:
  - Di tab **Live Simulation**, batas kapasitas per lajur diatur ke `hard_max_lane_capacity = 80` kendaraan/lajur (`640` kendaraan total maksimal) untuk menampung penambahan manual dan mode zoom-out `45%`–`125%`, sementara normalisasi observasi antrean `obs[0:8]` tetap menggunakan `queue_obs_norm_capacity = 30.0` dan dijepit ke `[0.0, 2.0]` agar kompatibel dengan model PPO yang dilatih pada kapasitas latihan standar (`30`/lajur).
- **Live Policy Preview (`TrainingJobManager._rollout_preview_frames`)**:
  - Selama training PPO berjalan *headless* di CPU pada 8 environment vektor (`SyncVectorEnv`), `TrainingJobManager` mengeksekusi 4 tick deterministik pada satu environment `rush_hour` terpisah setiap selesai 1 update PPO dan mengirim `preview_frames` lewat WebSocket `/ws/training`. Tab **Training Studio** memutar frame tersebut secara berkala tanpa memperlambat proses optimasi CPU.

---

## Hasil Benchmark Terbaru (10 Seed per Skenario, 163.840 Training Steps)

Tabel di bawah disinkronkan langsung dari `artifacts/benchmarks/benchmark_table.md` setelah integrasi model kolisi (`enable_collisions=True`, `all_red_steps=1`):

| Skenario | Agen | Parameter Terlatih | Mean Reward | Mean Queue | Mean Wait (s) | Mean Throughput | Mean Collisions | Latency (ms/step) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `normal` | Fixed Timer | - | -17.711 | 8.14 | 22.33 | 209.8 | 0.00 | 0.003 |
| `normal` | PPO-MLP | 69.755 | -10.582 | 4.97 | 13.79 | 211.0 | 0.00 | 0.273 |
| `normal` | PPO-GRU | 69.725 | -12.285 | 5.76 | 15.82 | 212.3 | 0.00 | 0.224 |
| `normal` | Fly Connectome PPO | 70.006 | -11.522 | 5.32 | 14.60 | 214.0 | 0.00 | 0.387 |
| `asymmetric` | Fixed Timer | - | -28.849 | 14.71 | 25.73 | 329.5 | 1.70 | 0.003 |
| `asymmetric` | PPO-MLP | 69.755 | -17.497 | 8.73 | 14.92 | 340.8 | 0.20 | 0.102 |
| `asymmetric` | PPO-GRU | 69.725 | -19.795 | 9.17 | 15.91 | 331.5 | 1.50 | 0.119 |
| `asymmetric` | Fly Connectome PPO | 70.006 | -16.432 | 8.35 | 14.60 | 333.9 | 0.00 | 0.369 |
| `rush_hour` | Fixed Timer | - | -29.093 | 14.90 | 25.21 | 327.9 | 1.60 | 0.003 |
| `rush_hour` | PPO-MLP | 69.755 | -18.440 | 9.29 | 16.02 | 337.1 | 0.50 | 0.093 |
| `rush_hour` | PPO-GRU | 69.725 | -20.243 | 9.86 | 17.20 | 329.9 | 0.80 | 0.126 |
| `rush_hour` | Fly Connectome PPO | 70.006 | -18.720 | 9.36 | 16.20 | 335.3 | 0.30 | 0.414 |

Uji tekanan *safety buffer* pada skenario `rush_hour` (`all_red_steps=0` / Yellow Only) menaikkan kolisi 10-seed `Fly Connectome PPO` dari `0.30` menjadi `9.60`, serta `Fixed Timer` dari `1.60` menjadi `4.60`.

---

## Dokumentasi Teknis

- [Panduan Training & Siklus Episode](docs/TRAINING_AND_EPISODE_GUIDE.md): Diagram Mermaid siklus 1 episode (`300` langkah), alur lengkap loop PPO CPU + Live Policy Preview, dan penelusuran episode nyata `rush_hour` beserta uji tekanan kolisi.
- [Kamus Variabel & Matematika](docs/VARIABLES_AND_MATH.md): Pemetaan simbol persamaan biologis, observasi, reward, kolisi, *safety buffer*, dan hyperparameter PPO ke nama variabel kode.
