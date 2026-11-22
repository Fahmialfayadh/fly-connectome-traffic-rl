# Drosophila Connectome RL Traffic Controller

Sistem kendali lampu lalu lintas berbasis *Reinforcement Learning* (PPO) yang menggantikan lapisan tersembunyi MLP/GRU standar dengan sub-graf biologis otak lalat buah (*Drosophila melanogaster*, dataset `FAFB v783`) berisi **2.500 neuron** dari sirkuit **Central Complex (`PB`, `FB`, `EB`, `NO`)** dan **Mushroom Body (`MB_*`)**, berjalan 100% pada CPU.

---

## Ringkasan Alur & Status Workspace

Proyek ini terdiri atas 5 fase yang sudah dapat dijalankan secara terpisah maupun interaktif lewat antarmuka web:

| Fase | Tujuan | Skrip / Antarmuka | Output Artefak |
|---|---|---|---|
| **Fase 1: Ekstraksi Graf** | Saring 1.800 neuron CX + 700 neuron MB dari `connections_princeton.csv`, pisahkan $W_{\text{exc}}$ dan $W_{\text{inh}}$, skalakan *spectral radius* ke `0.95`, serta bangun *null model* acak terkontrol (*global random rewiring*). | `scripts/extract_connectome.py`, `connectome/shuffled.py` | `artifacts/fly_subconnectome.pt`, `artifacts/fly_subconnectome_shuffled.pt` |
| **Fase 2 & 3: Training PPO** | Latih kebijakan `FlyConnectomePolicy` (`fly`), `Shuffled Connectome PPO` (`fly_shuffled`), `MLPBaselinePolicy`, dan `GRUBaselinePolicy` pada 8 environment CPU paralel menggunakan PPO + *decision-step masking* dan pantau **Live Policy Preview** (4 tick `rush_hour` per update). | **Tab Training Studio** di Web UI, atau `scripts/train_agents.py` | `artifacts/checkpoints/{fly,fly_shuffled,mlp,gru}_ppo.pt` |
| **Fase 4: Benchmark** | Evaluasi komparatif model yang dipilih pengguna (`Fixed Timer`, `PPO-MLP`, `PPO-GRU`, `Fly Connectome PPO`, `Shuffled Connectome PPO`, maupun *checkpoint* kustom `.pt`) di skenario `normal`, `asymmetric`, `rush_hour`, dan `stress_test` (termasuk grafik & metrik `mean_collisions`), serta uji farmakologi & lesi. | **Tab Benchmark & Connectome** di Web UI, atau `scripts/run_benchmark.py` | `artifacts/benchmarks/benchmark_summary.json`, `benchmark_table.md`, `*.png` |
| **Fase 5: Lab Simulasi, 3D & Dokumentasi** | Jalankan **Inference** secara *real-time* (60 FPS Canvas 2D + visualisasi 3D otak FlyWire + telemetri adaptif untuk model non-fly), inspeksi end-to-end arsitektur pada **Tab 3D Architecture** (*Forward Pass* vs *Backward Gradient Flow*), deteksi kolisi, *Safety Buffer*, kontrol $g_{\text{exc}} / g_{\text{inh}}$ + lesi, serta laporan teknis interaktif di **Tab Documentation**. | `scripts/launch_web_lab.py` (Utama) | Web UI di `http://127.0.0.1:8080` (`/simulation`, `/training`, `/benchmark`, `/architecture`, `/docs`) |

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
- **Tab Training Studio (`/training`)**: Pemilihan arsitektur kebijakan (`fly`, `fly_shuffled` *Null Model*, `mlp`, `gru`), fitur **Resume from Checkpoint** (otomatis menghitung sisa langkah `total_timesteps - last_model_step` dan melanjutkan optimizer/riwayat), pengaturan *timesteps* hingga **Mode Maximal** (`524.288` langkah) dan **Custom Timesteps** (`1.024`–`2.000.000`), panel **Custom PPO Hyperparameters**, pilihan **Model File Save Mode** (**Create a new model file (.pt)**, **Overwrite default**, **Overwrite resumed checkpoint**, atau **Dry run**), grafik optimasi *real-time*, inventaris *checkpoint* lokal dengan tombol **Load in Live Sim** & **Resume**, serta **Live Policy Preview** (`rush_hour`, 4 tick per update PPO via `/ws/training`).
- **Tab Benchmark & Connectome (`/benchmark`)**: Pemilih model interaktif (*checkbox* untuk memilih kombinasi model bawaan, `Shuffled Connectome PPO`, maupun *checkpoint* kustom `.pt` yang ingin dibandingkan via `POST /api/benchmark/run`), ringkasan sub-graf `FAFB v783`, grafik perbandingan multi-skenario (`normal`, `asymmetric`, `rush_hour`, `stress_test`), tabel hasil farmakologi/lesi, dan kamus variabel.
- **Tab 3D Architecture (`/architecture`)**: Visualisasi 3D interaktif *end-to-end* untuk `FlyConnectomePolicy` (`x_t -> input_projection -> ConnectomeRecurrentCell -> actor_head / critic_head -> environment`), sakelar aliran **Forward Pass** vs **Backward Gradient**, filter status (*Trainable*, *Frozen Bio*, *Runtime*), peta komponen horizontal, inspektor detail per modul, serta panduan matematis 4 tahap.
- **Tab Documentation (`/docs`)**: Laporan teknis lengkap berbahasa Indonesia dengan daftar isi *sticky*, persamaan matematis terformat, grafik Canvas interaktif tersinkronisasi dari `benchmark_summary.json`, daftar referensi dataset, serta glosari istilah.

### 2. Perintah CLI Opsional (Ekstraksi, Training, Resume, Benchmark, Tes)

```bash
# Fase 1: Ekstrak ulang sub-connectome dari dataset CSV
/cache/ml/.venv/bin/python scripts/extract_connectome.py

# Fase 2-3: Latih agen dari CLI (default --agents: fly mlp gru)
/cache/ml/.venv/bin/python scripts/train_agents.py --agents fly mlp gru --total-timesteps 163840

# Latih termasuk Null Model (fly_shuffled harus dideklarasikan di --agents)
/cache/ml/.venv/bin/python scripts/train_agents.py --agents fly fly_shuffled mlp gru --total-timesteps 163840 --parallel-models 4

# Resume training dari checkpoint yang sudah ada (sisa langkah = target - step terakhir)
/cache/ml/.venv/bin/python scripts/train_agents.py --agents fly fly_shuffled mlp gru --total-timesteps 327680 --resume

# Simpan atau resume dengan nama checkpoint ber-suffix khusus (mis. fly_ppo_exp2.pt)
/cache/ml/.venv/bin/python scripts/train_agents.py --agents fly --total-timesteps 163840 --suffix _exp2 --resume

# Fase 4: Evaluasi komparatif 10 seed per skenario (normal, asymmetric, rush_hour, stress_test)
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
├── frontend/                       # Antarmuka Web (Modular Partials + OKLCH CSS + ES6 + Canvas 2D + Three.js)
│   ├── index.html
│   ├── partials/                   # Potongan HTML per tab (simulation, training, benchmark, architecture, documentation)
│   ├── css/
│   │   ├── tokens.css
│   │   └── layout.css
│   └── js/
│       ├── main.js
│       ├── intersection_renderer.js
│       ├── connectome_renderer.js
│       ├── model_telemetry_renderer.js
│       ├── brain3d_renderer.js
│       ├── architecture_view.js
│       ├── training_studio.js
│       ├── benchmark_view.js
│       └── documentation_view.js
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
  - **`red_light_conflict`**: Lajur merah yang menunggu melebihi ambang `red_light_wait_threshold_s = 120.0` s berpeluang menerobos lampu merah (hingga `red_light_max_prob = 0.14`) dan bertabrakan dengan lajur hijau yang sedang melaju.
  - **`box_clearance_conflict`**: Saat terjadi perpindahan fase, lajur pelepas sebelumnya berpeluang `dilemma_zone_prob = 0.18` menyisakan kendaraan *straggler* di kotak persimpangan (`_box_stragglers`). Dengan *safety buffer* `all_red_steps = 1` (`Yellow + All-Red`), straggler terselesaikan sebelum fase baru menjadi `GREEN`; jika diubah menjadi `all_red_steps = 0` (`Yellow Only` lewat kontrol UI atau `set_safety_buffer(0)`), konflik kotak persimpangan langsung meningkat.
  - Setiap kolisi mengurangi *reward* langkah sebesar `collision_penalty * reward_scale` (`15.0 * 0.005 = 0.075`) dan dirender sebagai indikator bahaya pada Canvas 2D.
- **Kapasitas Live Simulation & Normalisasi Observasi (`SimulationService` / `TrafficIntersectionEnv`)**:
  - Di tab **Live Simulation**, batas kapasitas per lajur diatur ke `hard_max_lane_capacity = 80` kendaraan/lajur (`640` kendaraan total maksimal) untuk menampung penambahan manual dan mode zoom-out `45%`–`125%`, sementara normalisasi observasi antrean `obs[0:8]` tetap menggunakan `queue_obs_norm_capacity = 30.0` dan dijepit ke `[0.0, 2.0]` agar kompatibel dengan model PPO yang dilatih pada kapasitas latihan standar (`30`/lajur).
- **Live Policy Preview (`TrainingJobManager._rollout_preview_frames`)**:
  - Selama training PPO berjalan *headless* di CPU pada 8 environment vektor (`SyncVectorEnv`), `TrainingJobManager` mengeksekusi 4 tick deterministik pada satu environment `rush_hour` terpisah setiap selesai 1 update PPO dan mengirim `preview_frames` lewat WebSocket `/ws/training`. Tab **Training Studio** memutar frame tersebut secara berkala tanpa memperlambat proses optimasi CPU.

---

## Hasil Benchmark Terbaru (10 Seed per Skenario, 163.840 Training Steps)

Tabel di bawah disinkronkan langsung dari `artifacts/benchmarks/benchmark_table.md` setelah integrasi model kolisi (`enable_collisions=True`, `all_red_steps=1`, `red_light_wait_threshold_s=120.0`) serta profil beban ekstrem `stress_test`:

| Skenario | Agen | Parameter Terlatih | Mean Reward | Mean Queue | Mean Wait (s) | Mean Throughput | Mean Collisions | Latency (ms/step) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `normal` | Fixed Timer | - | -17.688 | 8.06 | 22.33 | 208.2 | 0.00 | 0.002 |
| `normal` | PPO-MLP | 69.755 | -10.731 | 5.03 | 13.90 | 211.9 | 0.00 | 0.075 |
| `normal` | PPO-GRU | 69.725 | -12.010 | 5.62 | 15.50 | 212.7 | 0.00 | 0.165 |
| `normal` | Fly Connectome PPO | 70.006 | -11.477 | 5.29 | 14.50 | 213.6 | 0.00 | 0.189 |
| `asymmetric` | Fixed Timer | - | -31.220 | 15.93 | 28.14 | 328.0 | 0.00 | 0.001 |
| `asymmetric` | PPO-MLP | 69.755 | -16.544 | 8.27 | 14.30 | 337.3 | 0.00 | 0.068 |
| `asymmetric` | PPO-GRU | 69.725 | -19.962 | 9.21 | 15.88 | 338.5 | 0.00 | 0.077 |
| `asymmetric` | Fly Connectome PPO | 70.006 | -16.792 | 8.55 | 15.09 | 332.5 | 0.00 | 0.198 |
| `rush_hour` | Fixed Timer | - | -29.838 | 15.26 | 26.29 | 324.7 | 0.00 | 0.001 |
| `rush_hour` | PPO-MLP | 69.755 | -18.950 | 9.55 | 16.65 | 329.8 | 0.00 | 0.065 |
| `rush_hour` | PPO-GRU | 69.725 | -22.309 | 10.85 | 18.35 | 339.5 | 0.00 | 0.106 |
| `rush_hour` | Fly Connectome PPO | 70.006 | -18.269 | 9.16 | 16.02 | 332.3 | 0.00 | 0.473 |
| `stress_test` | Fixed Timer | - | -241.743 | 140.88 | 85.60 | 770.6 | 42.50 | 0.005 |
| `stress_test` | PPO-MLP | 69.755 | -317.427 | 185.54 | 136.41 | 546.7 | 41.90 | 0.182 |
| `stress_test` | PPO-GRU | 69.725 | -272.365 | 150.45 | 116.43 | 544.7 | 55.60 | 0.299 |
| `stress_test` | Fly Connectome PPO | 70.006 | -320.557 | 191.08 | 143.97 | 521.5 | 42.80 | 0.486 |

---

## Dokumentasi Teknis

- [Laporan Teknis Utama](docs/MASTER_DOC.md): Pembahasan menyeluruh konstruksi sub-konektom FAFB v783, *Shuffled Null Model*, lingkungan lalu lintas, arsitektur kebijakan, PPO, serta hasil evaluasi.
- [Panduan Training & Siklus Episode](docs/TRAINING_AND_EPISODE_GUIDE.md): Diagram Mermaid siklus 1 episode (`300` langkah), alur lengkap loop PPO CPU + Live Policy Preview, dan penelusuran episode nyata `rush_hour` beserta uji tekanan kolisi.
- [Kamus Variabel & Matematika](docs/VARIABLES_AND_MATH.md): Pemetaan simbol persamaan biologis, observasi, reward, kolisi, *safety buffer*, dan hyperparameter PPO ke nama variabel kode.
