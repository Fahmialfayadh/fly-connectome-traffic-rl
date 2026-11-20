# Kamus Variabel, Arsitektur OOP, dan Pemetaan Matematika

Dokumen ini memetakan seluruh simbol persamaan di `PLAN.md` beserta parameter algoritma PPO, mekanisme kolisi, dan *safety buffer* ke nama file, kelas, dan variabel aktual di dalam kode Python. Untuk diagram alur visual dan penelusuran 1 episode penuh, lihat [TRAINING_AND_EPISODE_GUIDE.md](TRAINING_AND_EPISODE_GUIDE.md).

---

## 1. Pemetaan Persamaan Biologis & RL ke Variabel Kode

### A. Modul Connectome (`src/fly_traffic/connectome/`)

| Simbol / Konsep di `PLAN.md` | Nama Variabel / Atribut di Kode | Lokasi File & Kelas | Keterangan Fungsional |
|---|---|---|---|
| Sirkuit Central Complex | `ConnectomeConfig.cx_neuropils` | `src/fly_traffic/config.py` | Tuple neuropil target: `("PB", "FB", "EB", "NO")`, kuota `1.800` neuron (`cx_neuron_quota`). |
| Sirkuit Mushroom Body | `ConnectomeConfig.mb_neuropils` | `src/fly_traffic/config.py` | Tuple 8 lobus/kaliks MB (`MB_CA_*`, `MB_ML_*`, `MB_PED_*`, `MB_VL_*`), kuota `700` neuron (`mb_neuron_quota`). |
| $\log(1 + \text{syn\_count})$ | `synapse_weights = np.log1p(...)` | `preprocessor.py` (`ConnectomePreprocessor._build_polarity_csr`) | Kompresi logaritmik jumlah sinapsis teragregasi per pasangan `(post_root_id, pre_root_id)`. |
| $W_{\text{exc}}$ (Eksitatori / ACh) | `w_exc_csr` (buffer sparse CSR) | `artifact.py`, `connectome_cell.py` (`ConnectomeRecurrentCell`) | Matriks konektivitas sinapsis tipe `ACH` dengan `requires_grad=False` (dikunci/frozen). |
| $W_{\text{inh}}$ (Inhibitori / GABA + GLUT) | `w_inh_csr` (buffer sparse CSR) | `artifact.py`, `connectome_cell.py` (`ConnectomeRecurrentCell`) | Matriks konektivitas sinapsis tipe `GABA` dan `GLUT` dengan `requires_grad=False` (dikunci/frozen). |
| *Spectral radius* awal & target | `raw_spectral_radius`, `target_spectral_radius` (`0.95`) | `preprocessor.py` (`_compute_spectral_radius`) | Nilai eigen terbesar $\rho(W_{\text{exc}} - W_{\text{inh}})$ sebelum (`113.1494`) dan sesudah (`0.9500`) penskalaan. |

### B. Modul Dinamika Sel Saraf Lalat (`src/fly_traffic/agents/connectome_cell.py` & `fly_policy.py`)

Persamaan inti:
$$h_{t+1} = (1 - \tau) h_t + \tau \tanh\Big((g_{\text{exc}} W_{\text{exc}} - g_{\text{inh}} W_{\text{inh}}) h_t + W_{\text{in}} x_t\Big)$$

| Simbol Persamaan | Nama Variabel / Properti di Kode | Status Optimasi | Keterangan |
|---|---|---|---|
| $h_t$ | `hidden_state` (shape `[batch, 2500]`) | State dinamis | Vektor aktivasi seluruh 2.500 neuron (indeks `0..1799` = CX, `1800..2499` = MB). |
| $\tau$ (*leak rate*) | `leak_rate` (dihitung dari `tau_logit`) | **Trainable** (`1` param) | Dibatasi ke $(0.05, 0.95)$ lewat `0.05 + 0.90 * torch.sigmoid(self.tau_logit)`. |
| $g_{\text{exc}}$ | `self.g_exc` (default `1.0`) | Kontrol *real-time* | Pengali gain sinapsis eksitatori (`ACh`) yang dapat digeser langsung via slider `[0.0 .. 2.5]`. |
| $g_{\text{inh}}$ | `self.g_inh` (default `1.0`) | Kontrol *real-time* | Pengali gain sinapsis inhibitori (`GABA`/`GLUT`) yang dapat digeser langsung via slider `[0.0 .. 2.5]`. |
| $W_{\text{in}} x_t$ | `self.input_projection(obs)` | **Trainable** (`57.500` param) | Proyeksi linear dari vektor observasi `22`-dim ke `2.500` neuron. |
| $W_{\text{out}}$ (Policy & Value heads) | `self.actor_head`, `self.critic_head` | **Trainable** (`12.505` param) | `actor_head`: `2500 -> 4` logits fase lampu; `critic_head`: `2500 -> 1` estimasi nilai $V(s_t)$. |
| Masker Lesi (*In-Silico Ablation*) | `self.lesion_mask` (buffer `[1, 2500]`) | Kontrol *real-time* | Bernilai `1.0` untuk neuron aktif dan `0.0` untuk neuron yang dibungkam (*lesioned*). |

### C. Modul Traffic Environment (`src/fly_traffic/environment/`)

Persamaan *reward* aktual (`TrafficIntersectionEnv._compute_reward`):
$$r_t = \text{reward\_scale} \cdot \left(-\sum \text{queue\_length} - \alpha \cdot \text{max\_wait\_time} - \beta \cdot \mathbb{I}(\text{phase\_switch}) - \text{collision\_penalty} \cdot \text{step\_collisions}\right)$$

| Simbol / Elemen | Nama Variabel di Kode | Nilai Default | Lokasi File |
|---|---|---|---|
| $x_t$ (*Observation Space*) | `observation` (`Box(0, 2, shape=(22,))`) | 22 float32 | `traffic_env.py` (`_build_observation`) |
| Indeks `obs[0:8]` | Panjang antrean ternormalisasi (`lane.length / 30`) | `[0.0 .. 1.0]` | `LaneId` (`N_STRAIGHT .. W_TURN`) |
| Indeks `obs[8:16]` | Waktu tunggu terlama per lajur (`max_wait_s / 120.0`) | `[0.0 .. 2.0]` | `LaneQueue.max_wait_seconds` |
| Indeks `obs[16:20]` | *One-hot* fase lampu aktif | 4 bit | `SignalPhase` (`0: NS Straight, 1: NS Turn, 2: EW Straight, 3: EW Turn`) |
| Indeks `obs[20]` | Flag transisi kuning / all-red (`is_transitioning`) | `0.0` atau `1.0` | `SignalController.is_transitioning` |
| Indeks `obs[21]` | Durasi fase berjalan ternormalisasi (`phase_elapsed / 20.0`) | `[0.0 .. 2.0]` | `SignalController.phase_elapsed` |
| $a_t$ (*Action Space*) | `target_phase` (`Discrete(4)`) | `0..3` | `TrafficIntersectionEnv.step` |
| $\alpha$ | `wait_penalty_alpha` | `0.1` | `TrafficEnvConfig` |
| $\beta$ | `switch_penalty_beta` | `1.0` | `TrafficEnvConfig` |
| Skala reward | `reward_scale` | `0.005` | `TrafficEnvConfig` |
| Batas keamanan sinyal | `min_green_steps`, `yellow_steps`, `all_red_steps` | `2`, `1`, `1` langkah (`dt=2.0s`) | `SignalController` (`intersection_state.py`) & `TrafficEnvConfig` |

### D. Parameter Kolisi & *Safety Buffer* (`src/fly_traffic/environment/traffic_env.py` & `config.py`)

Lingkungan memodelkan dua jenis konflik lalu lintas ketika `enable_collisions = True`:

1. **`red_light_conflict` (*Starvation-Induced Red-Light Violation*, `_evaluate_red_light_conflicts`)**:
   Saat `SignalStage.GREEN` dan terdapat minimal satu lajur hijau yang melepas kendaraan (`discharging_lanes`), setiap lajur merah berantrean dengan waktu tunggu kepala antrean `head_wait_s > red_light_wait_threshold_s` berpeluang menerobos lampu merah sebesar:
   $$p_{\text{red}} = \min\left(\text{red\_light\_max\_prob},\ \frac{\text{head\_wait\_s} - \text{red\_light\_wait\_threshold\_s}}{140.0} \cdot \text{red\_light\_max\_prob}\right)$$
   Jika terpicu, 1 kendaraan dikeluarkan dari lajur pelanggar dan tercatat sebagai `red_light_conflict`.
2. **`box_clearance_conflict` (*Dilemma-Zone Intergreen Straggler*, `_evaluate_box_clearance_conflicts`)**:
   Ketika `phase_switched = True`, tiap lajur yang baru melepas pada langkah sebelumnya (`previous_departures > 0`) berpeluang `dilemma_zone_prob` menyisakan kendaraan tertinggal di kotak persimpangan (`self._box_stragglers` dengan sisa umur `remaining_steps = 2`). Jika `remaining_steps > 0` bertepatan dengan lajur dari fase berbeda yang aktif melepas kendaraan (`discharging_lanes`), tercatat `box_clearance_conflict`. Dengan `yellow_steps = 1` dan `all_red_steps >= 1`, dua langkah jeda tanpa pelepasan menghabiskan sisa umur `2 -> 1 -> 0` sebelum fase baru mendapat `GREEN`; bila `all_red_steps = 0`, fase baru sudah aktif `GREEN` pada `remaining_steps = 1` sehingga berisiko memicu konflik kotak persimpangan.

| Parameter / Variabel | Nilai Default / Tipe | Lokasi File & Metode | Keterangan Fungsional |
|---|---|---|---|
| `collision_penalty` | `15.0` | `TrafficEnvConfig` / `_compute_reward` | Penalti per event kolisi sebelum dikalikan `reward_scale = 0.005` (setara pengurangan `-0.075` pada reward langkah per kolisi). |
| `red_light_wait_threshold_s` | `36.0` (detik) | `TrafficEnvConfig` / `_evaluate_red_light_conflicts` | Ambang `head_wait_s` pada lajur merah sebelum probabilitas pelanggaran lampu merah mulai naik di atas `0.0`. |
| `red_light_max_prob` | `0.14` | `TrafficEnvConfig` / `_evaluate_red_light_conflicts` | Batas atas probabilitas per langkah per lajur merah untuk memicu `red_light_conflict`. |
| `dilemma_zone_prob` | `0.18` | `TrafficEnvConfig` / `_evaluate_box_clearance_conflicts` | Probabilitas per lajur pelepas sebelumnya menyisakan *straggler* kotak persimpangan saat terjadi perpindahan fase. |
| `enable_collisions` | `True` | `TrafficEnvConfig` / `_evaluate_collisions` | Sakelar pengaktif atau penonaktif evaluasi kedua jenis kolisi. |
| `all_red_steps` / `set_safety_buffer` | `1` (dapat diubah `>= 0`) | `TrafficEnvConfig`, `TrafficIntersectionEnv.set_safety_buffer`, `SimulationService.set_safety_buffer` | Durasi tahap `SignalStage.ALL_RED` dalam langkah (`dt = 2.0` s); nilai `0` mengaktifkan skenario stres tanpa jeda *all-red*. |
| `total_collisions` | `int` (`>= 0`) | `TrafficIntersectionEnv._total_collisions`, `info`, `get_render_snapshot`, `EpisodeMetrics` | Jumlah akumulatif event kolisi dalam 1 episode; diagregasi sebagai `mean_collisions` di `ScenarioBenchmarkResult` dan `PharmacologyResult` (`evaluator.py`). |
| `step_collisions` & `step_collision_events` | `int` & `list[dict]` | `info` & `get_render_snapshot()` | Jumlah kolisi pada langkah berjalan dan daftar metadata `{step, kind, culprit_lane, victim_lane, culprit_wait_s}` untuk rendering peringatan 2D (`IntersectionRenderer._processCollisionEvents`). |

---

## 2. Alur Transparan Training PPO (`src/fly_traffic/training/`)

1. **Vector Rollout Collection (`PPOTrainer.train`)**:
   - Menjalankan `num_envs = 8` instance `TrafficIntersectionEnv` (profil `mixed`: sampel acak `normal`, `asymmetric`, `asymmetric_ew`, atau `rush_hour` dengan skala laju `Uniform(0.85, 1.15)`) secara paralel dengan mode `AutoresetMode.SAME_STEP`.
   - Setiap update mengumpulkan `rollout_steps = 128` langkah per env (`batch_steps = 1024` langkah total).
   - Pada awal episode baru (`previous_done_mask == 1.0`), state rekuren $h_t$ di-nol-kan kembali agar memori antar-episode tidak bocor.
   - `decision_mask` bernilai `1.0` hanya ketika lampu berstatus `GREEN` (`obs[:, 20] < 0.5`) dan `phase_elapsed >= min_green_steps`, sehingga gradien aktor dan entropi tidak terdilusi oleh langkah *no-op* selama fase `YELLOW`/`ALL_RED`.
2. **Advantage Estimation (`RecurrentRolloutBuffer.compute_advantages`)**:
   - Menghitung `temporal_difference_error` ($\delta_t$) dan mengakumulasi `running_gae_advantage` dengan $\gamma = 0.99$ (`gamma`) dan $\lambda_{\text{GAE}} = 0.95$ (`gae_lambda`).
3. **Sequence-Chunked Optimization (`PPOTrainer._ppo_update`)**:
   - Memecah rollout `128 x 8` menjadi potongan sekuens panjang `sequence_chunk_len = 16` (`64` chunk), lalu memperbarui bobot sebanyak `update_epochs = 4` epoch menggunakan `num_minibatches = 4` minibatch acak per epoch (`16` chunk atau `256` langkah per minibatch).
   - Menghitung `policy_loss` (PPO clipped surrogate, $\epsilon = 0.2$), `value_loss` (MSE `0.5 * (V - R)^2`, koefisien `0.25`), dan `entropy_loss` (koefisien `0.005`), dengan `clip_grad_norm_(0.5)` dan peluruhan *learning rate* linear dari `8e-4` hingga `1.2e-4`.
4. **Live Policy Preview & Penyimpanan Otomatis (`TrainingJobManager`)**:
   - Pada mode Training Studio di Web UI, `PPOTrainer.train` memanggil `on_policy_update_callback(policy, metrics_dict)` setelah tiap update selesai. `TrainingJobManager._rollout_preview_frames` menjalankan 4 tick deterministik pada satu `preview_env` (`rush_hour`) dan mengirim `preview_frames` melalui WebSocket `/ws/training` ke `TrainingStudioView`.
   - Setelah seluruh update selesai (dan tidak dibatalkan), bobot disimpan ke `artifacts/checkpoints/<model_type>_ppo.pt` beserta riwayat metrik `TrainingHistory` dan metadata `trained_timesteps`, lalu dimuat ulang secara otomatis ke `SimulationService`.
