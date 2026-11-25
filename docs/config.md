# Referensi Konfigurasi (`src/fly_traffic/config.py`)

Seluruh konfigurasi terpusat di [`src/fly_traffic/config.py`](../src/fly_traffic/config.py) dalam lima `@dataclass(frozen=True)` dan tiga konstanta path agar sifatnya *immutable* dan konsisten di seluruh modul ekstraksi, environment, training PPO, benchmark, maupun UI.

---

## 1. Konstanta Path Global

| Nama | Nilai Default | Keterangan |
|---|---|---|
| `PROJECT_ROOT` | `Path(__file__).resolve().parents[2]` | Direktori root repositori yang diturunkan relatif dari lokasi `config.py`. |
| `DEFAULT_DATASET_PATH` | `PROJECT_ROOT / "dataset" / "connections_princeton.csv"` | Lokasi default tabel koneksi mentah FlyWire FAFB v783. |
| `DEFAULT_ARTIFACTS_DIR` | `PROJECT_ROOT / "artifacts"` | Direktori induk tempat menyimpan artefak graf, checkpoint model, dan hasil benchmark. |

---

## 2. `ConnectomeConfig`

Mengatur seleksi sub-konektom otak lalat (Central Complex + Mushroom Body), pemisahan polaritas sinaptik, dan penskalaan spektral.

| Field | Tipe | Default | Keterangan |
|---|---|---|---|
| `dataset_path` | `Path` | `DEFAULT_DATASET_PATH` | Path ke berkas CSV koneksi sinapsis mentah FAFB v783. |
| `output_path` | `Path` | `DEFAULT_ARTIFACTS_DIR / "fly_subconnectome.pt"` | Lokasi keluaran artefak sub-konektom PyTorch. |
| `cx_neuropils` | `tuple[str, ...]` | `("PB", "FB", "EB", "NO")` | Target neuropil pada sirkuit *Central Complex* (navigasi & pemilihan aksi). |
| `mb_neuropils` | `tuple[str, ...]` | `("MB_CA_L", "MB_CA_R", "MB_ML_L", "MB_ML_R", "MB_PED_L", "MB_PED_R", "MB_VL_L", "MB_VL_R")` | Delapan neuropil kiri/kanan pada sirkuit *Mushroom Body* (memori asosiatif). |
| `cx_neuron_quota` | `int` | `1800` | Jumlah neuron teraktif (berdasarkan total `syn_count`) yang diambil dari CX. |
| `mb_neuron_quota` | `int` | `700` | Jumlah neuron teraktif yang diambil dari MB (tanpa duplikasi neuron CX). |
| `excitatory_nts` | `tuple[str, ...]` | `("ACH",)` | Neurotransmiter yang dipetakan ke matriks bobot eksitatori $W_{\text{exc}}$. |
| `inhibitory_nts` | `tuple[str, ...]` | `("GABA", "GLUT")` | Neurotransmiter yang dipetakan ke matriks bobot inhibitori $W_{\text{inh}}$. |
| `target_spectral_radius` | `float` | `0.95` | Nilai eigen terbesar target untuk matriks bersih $(W_{\text{exc}} - W_{\text{inh}})$ agar dinamika rekuren tidak meledak. |
| `power_iteration_seed` | `int` | `42` | Seed vektor awal untuk estimasi nilai eigen Arnoldi / *power iteration*. |

---

## 3. `TrafficEnvConfig`

Mengatur fisika antrean diskrit, kapasitas pelepasan, aturan *interlock* sinyal, pembobotan *reward*, dan mekanisme kolisi pada `TrafficIntersectionEnv`.

| Field | Tipe | Default | Keterangan |
|---|---|---|---|
| `dt_seconds` | `float` | `2.0` | Durasi waktu simulasi nyata per satu langkah diskrit (`step`). |
| `max_steps` | `int` | `500` | Panjang satu episode tetap (`500` langkah = `1.000` detik simulasi). |
| `max_lane_capacity` | `int` | `30` | Kapasitas antrean per lajur bawaan selama pelatihan dan evaluasi. |
| `queue_obs_norm_capacity` | `float` | `30.0` | Pembagi normalisasi antrean ke fitur `obs[0:8]` (dijepit ke `[0.0, 2.0]`). |
| `hard_max_lane_capacity` | `int` | `80` | Batas maksimal kapasitas per lajur (`640` total) untuk suntikan kendaraan manual di Live Simulation. |
| `straight_discharge_rate` | `float` | `2.5` | Laju pelepasan kendaraan per step hijau pada lajur lurus. |
| `turn_discharge_rate` | `float` | `2.0` | Laju pelepasan kendaraan per step hijau pada lajur belok. |
| `min_green_steps` | `int` | `6` | Durasi hijau minimum (`12,0 s`) sebelum sinyal diizinkan berpindah fase. |
| `yellow_steps` | `int` | `2` | Durasi tahap transisi kuning (`4,0 s`) saat perpindahan fase. |
| `all_red_steps` | `int` | `1` | Durasi jeda merah seluruh arah (`2,0 s`) setelah kuning sebelum fase baru hijau (*Safety Buffer*). |
| `wait_penalty_alpha` | `float` | `0.1` | Bobot penalti `max_wait_s` di dalam rumus *reward*. |
| `switch_penalty_beta` | `float` | `1.0` | Penalti tambahan setiap kali pergantian fase dipicu. |
| `reward_scale` | `float` | `0.005` | Faktor pengali skala akhir *reward* per langkah. |
| `max_wait_norm_seconds` | `float` | `150.0` | Pembagi normalisasi waktu tunggu kepala antrean ke fitur `obs[8:16]`. |
| `max_elapsed_norm_steps` | `float` | `20.0` | Pembagi normalisasi durasi hijau berjalan ke fitur `obs[21]`. |
| `profile_name` | `str` | `"mixed"` | Profil kedatangan kendaraan bawaan (`mixed` mengacak skenario tiap episode latih). |
| `collision_penalty` | `float` | `80.0` | Penalti mentah per kejadian kolisi (setara `-0,400` setelah dikali `reward_scale`). |
| `red_light_wait_threshold_s` | `float` | `150.0` | Ambang waktu tunggu merah sebelum muncul peluang pelanggaran menerobos (`red_light_conflict`). |
| `red_light_max_prob` | `float` | `0.14` | Batas maksimum probabilitas pelanggaran lampu merah per step pada lajur yang melewati ambang tunggu. |
| `dilemma_zone_prob` | `float` | `0.18` | Peluang tersisanya kendaraan *straggler* di tengah simpang saat fase baru berganti (`box_clearance_conflict`). |
| `enable_collisions` | `bool` | `True` | Mengaktifkan atau menonaktifkan evaluasi kejadian kolisi. |

---

## 4. `PPOConfig`

Mengatur hyperparameter algoritma *Proximal Policy Optimization* (PPO) beserta pelatihan rekuren BPTT per potongan urutan.

| Field | Tipe | Default | Keterangan |
|---|---|---|---|
| `total_timesteps` | `int` | `163840` | Total target langkah environment per sesi pelatihan default (`160` update). |
| `num_envs` | `int` | `8` | Jumlah `SyncVectorEnv` paralel per model saat mengumpulkan *rollout*. |
| `rollout_steps` | `int` | `128` | Jumlah langkah per environment sebelum satu update PPO (`8 × 128 = 1.024` transisi/update). |
| `sequence_chunk_len` | `int` | `16` | Panjang potongan urutan untuk BPTT pada jaringan rekuren (`128 / 16 = 8` *chunk* per env). |
| `num_minibatches` | `int` | `4` | Jumlah pembagian *minibatch* per epoch update. |
| `update_epochs` | `int` | `4` | Jumlah *epoch* optimasi di atas satu batch *rollout*. |
| `learning_rate` | `float` | `8e-4` | Laju belajar awal optimizer Adam (`eps=1e-5`). |
| `gamma` | `float` | `0.99` | Faktor diskon *reward* masa depan. |
| `gae_lambda` | `float` | `0.95` | Parameter *Generalized Advantage Estimation* (GAE). |
| `clip_coef` | `float` | `0.2` | Batas *clipping* rasio kebijakan pada *surrogate loss* PPO. |
| `ent_coef` | `float` | `0.005` | Koefisien bonus entropi untuk menjaga eksplorasi aksi. |
| `vf_coef` | `float` | `0.25` | Bobot *value loss* (`critic_head`) dalam total loss PPO. |
| `max_grad_norm` | `float` | `0.5` | Batas maksimum norma gradien (*gradient clipping*). |
| `anneal_lr` | `bool` | `True` | Meluruhkan `learning_rate` secara linear menuju akhir training. |
| `seed` | `int` | `42` | Seed acak untuk reproduksibilitas bobot dan environment. |
| `num_cpu_threads` | `int` | `4` | Jumlah thread CPU PyTorch dalam pelatihan standar. |
| `checkpoints_dir` | `Path` | `DEFAULT_ARTIFACTS_DIR / "checkpoints"` | Direktori penyimpanan berkas checkpoint model `.pt`. |

---

## 5. `BenchmarkConfig`

Mengatur evaluasi deterministik lintas skenario dan pengukuran latensi inferensi CPU.

| Field | Tipe | Default | Keterangan |
|---|---|---|---|
| `scenarios` | `tuple[str, ...]` | `("normal", "asymmetric", "rush_hour", "stress_test")` | Daftar skenario uji yang dijalankan pada evaluasi penuh. |
| `eval_seeds` | `tuple[int, ...]` | `(1000..1049)` | Lima puluh seed deterministik identik per skenario. |
| `latency_warmup_steps` | `int` | `20` | Jumlah langkah pemanasan sebelum waktu inferensi CPU dicatat. |
| `latency_measure_steps` | `int` | `200` | Jumlah langkah inferensi yang diukur rata-rata waktunya (`ms/step`). |
| `output_dir` | `Path` | `DEFAULT_ARTIFACTS_DIR / "benchmarks"` | Direktori keluaran laporan JSON, tabel Markdown, dan gambar PNG. |

---

## 6. `SandboxUIConfig`

Mengatur tampilan aplikasi Pygame opsional (`scripts/launch_sandbox.py`) serta batas default kontrol farmakologi dan lesi.

| Field | Tipe | Default | Keterangan |
|---|---|---|---|
| `window_width` | `int` | `1280` | Lebar jendela Pygame (piksel). |
| `window_height` | `int` | `760` | Tinggi jendela Pygame (piksel). |
| `left_panel_width` | `int` | `680` | Lebar panel simulasi simpang di sisi kiri. |
| `right_panel_width` | `int` | `600` | Lebar panel telemetri otak di sisi kanan. |
| `target_fps` | `int` | `60` | Kecepatan render visual jendela Pygame. |
| `sim_steps_per_second` | `float` | `5.0` | Kecepatan langkah simulasi bawaan di sandbox Pygame. |
| `heatmap_display_size` | `int` | `360` | Ukuran sisi kanvas heatmap aktivasi neuron (piksel). |
| `heatmap_grid_side` | `int` | `50` | Ukuran kisi 2D `50 × 50` untuk memetakan `2.500` neuron. |
| `gain_min` | `float` | `0.0` | Batas bawah pengali sinaptik $g_{\text{exc}}$ dan $g_{\text{inh}}$. |
| `gain_max` | `float` | `2.5` | Batas atas pengali sinaptik $g_{\text{exc}}$ dan $g_{\text{inh}}$. |
| `lesion_fraction` | `float` | `0.25` | Proporsi neuron yang dinonaktifkan (`25%`) per aksi lesi sirkuit. |
| `window_title` | `str` | `"Drosophila Connectome Traffic Control Sandbox"` | Judul jendela aplikasi Pygame. |
