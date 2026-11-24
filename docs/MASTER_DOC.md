# Dapatkah Lalat Mengatur Lalu Lintas?

Laporan teknis tentang pengujian kelayakan sub-konektom *Drosophila melanogaster* sebagai inti rekuren (*recurrent core*) pada kebijakan pengendalian lampu lalu lintas persimpangan empat arah.

---

## Pendahuluan: Pertanyaan Experiment

Proyek ini berangkat dari pertanyaan mendasar:

> *"Apakah sub-graf saraf biologis otak lalat buah, yang diekstrak dari data konektom nyata, dapat berfungsi sebagai inti rekuren dalam kebijakan pengendalian lampu lalu lintas pada persimpangan empat arah?"*

Pertanyaan ini sengaja dirumuskan sebagai uji kemampuan (*capability question*), bukan klaim keunggulan atas pengendali konvensional. Fokus utamanya adalah memeriksa apakah topologi sinaptik biologis—beserta pemisahan polaritas eksitatori-inhibitori dan dinamika memori rekuren—dapat mendukung pembelajaran penguat (*reinforcement learning*), sekaligus memungkinkan manipulasi farmakologis dan lesi sirkuit secara langsung selama simulasi berjalan.

---

## Bagian I: Konstruksi Sub-Konektom Lalat

Sebelum dipakai sebagai inti kebijakan, graf saraf harus diekstrak dari data mentah, dipisahkan berdasarkan polaritas neurotransmiter, dan diskalakan agar dinamika rekurennya tetap stabil.

### Sumber Data

Data konektom berasal dari dataset **FlyWire FAFB v783** melalui berkas `dataset/connections_princeton.csv` (`261 MiB` tidak terkompresi; `66 MiB` dalam arsip `.csv.gz`). Tabel ini mencatat setiap pasangan neuron pengirim (`pre_root_id`) dan penerima (`post_root_id`), jumlah sinaps (`syn_count`), tipe neurotransmiter (`nt_type`), serta neuropil lokasi sinaps (`neuropil`).

### Seleksi Neuron dan Sirkuit Target

Sub-graf dipilih dari dua wilayah anatomis utama pada konfigurasi `ConnectomeConfig` (`src/fly_traffic/config.py`):

1. **Central Complex (CX)** — mencakup neuropil `PB`, `FB`, `EB`, dan `NO`, yang berperan dalam seleksi aksi dan navigasi arah. Kuota targetnya adalah 1.800 neuron.
2. **Mushroom Body (MB)** — mencakup delapan neuropil kiri dan kanan (`MB_CA_L/R`, `MB_ML_L/R`, `MB_PED_L/R`, dan `MB_VL_L/R`), yang berperan dalam memori asosiatif dan konteks. Kuota targetnya adalah 700 neuron.

Pada masing-masing wilayah, neuron diurutkan berdasarkan total sinaps insiden (`syn_count` masuk ditambah keluar) lalu diambil sebanyak kuota. Neuron yang sudah masuk ke himpunan CX dikecualikan dari seleksi MB agar tidak terjadi duplikasi. Setelah 2.500 neuron terpilih, sisi-sisi sinaptik diinduksi di seluruh neuropil sehingga koneksi lintas wilayah (*cross-region edges*) antara CX dan MB tetap terjaga.

Hasil ekstraksi yang tersimpan di `artifacts/fly_subconnectome.pt` memuat:

- **Total neuron**: 2.500 (1.800 CX + 700 MB)
- **Sisi eksitatori**: 36.908
- **Sisi inhibitori**: 50.263
- **Koneksi lintas wilayah (CX ↔ MB)**: 2.880 baris sinaptik
- **Tingkat kejarangan (*sparsity*)**: `98,6053%` (artinya `98,6053%` dari `6.250.000` pasangan potensial bernilai nol, dan hanya `1,3947%` yang memiliki koneksi non-nol)

Format penyimpanan *Compressed Sparse Row* (CSR) memanfaatkan tingkat kejarangan ini agar perkalian matriks hanya memproses sisi yang aktif.

### Pemisahan Polaritas dan Pembobotan Sinaptik

Berdasarkan kolom `nt_type`, koneksi dipisahkan ke dalam dua matriks ketetanggaan terpisah:

- $W_{\text{exc}}$ — sinaps eksitatori (`ACH` / asetilkolin) yang memperkuat aktivasi neuron penerima.
- $W_{\text{inh}}$ — sinaps inhibitori (`GABA` dan `GLUT`) yang menekan aktivasi neuron penerima.

Bobot setiap pasangan terarah `(post, pre)` dihitung dengan kompresi logaritmik:

$$w_{ij} = \log(1 + \text{syn\_count}_{ij})$$

Pendekatan *log1p* mencegah pasangan dengan ribuan sinaps mendominasi jaringan secara ekstrem, sementara koneksi berjumlah sinaps kecil tetap berkontribusi secara proporsional.

### Normalisasi *Spectral Radius*

Matriks gabungan mentah $W_{\text{exc}} - W_{\text{inh}}$ memiliki *spectral radius* (magnitudo nilai eigen terbesar) sebesar `113,1494`. Nilai di atas `1,0` berisiko membuat aktivasi rekuren meledak (*explode*) seiring bertambahnya langkah waktu. Oleh karena itu, kedua matriks dikalikan faktor penskalaan yang sama hingga *spectral radius* bersihnya turun menjadi `0,9500`.

```python
# File: src/fly_traffic/connectome/preprocessor.py
# Potongan dari ConnectomePreprocessor.extract() (baris 98-110)

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

Setelah penskalaan, matriks dikonversi ke tensor koordinat (COO) dan diserialisasi ke `artifacts/fly_subconnectome.pt` bersama daftar `neuron_ids`, label `circuit_groups`, `dominant_neuropils`, `dominant_nts`, dan metadata ekstraksi. Dengan begitu, proses training tidak perlu memindai ulang berkas CSV mentah setiap kali dijalankan.

### Model Nol Terkontrol (*Shuffled Connectome Null Model*)

Untuk mengisolasi kontribusi **topologi biologis otak lalat** dari sekadar efek proyeksi ke ruang rekuren berdimensi tinggi (*reservoir computing*), modul `src/fly_traffic/connectome/shuffled.py` menyediakan pembanding terkontrol `artifacts/fly_subconnectome_shuffled.pt` (*global uniform random edge rewiring*, `seed = 42`):

- **Ukuran dan kepadatan identik**: tetap `2.500` neuron (`CX=1800`, `MB=700`) dengan *sparsity* persis `98,6053%`.
- **Distribusi bobot dan polaritas identik**: seluruh nilai bobot dari `36.908` sisi eksitatori dan `50.263` sisi inhibitori dipertahankan, lalu dipasangkan ulang secara acak seragam tanpa duplikasi ke pasangan `(post, pre)` di seluruh matriks `2.500 × 2.500`. Langkah ini menghancurkan modularitas biologis (koneksi lintas wilayah `CX ↔ MB` meningkat dari `2.880` menjadi `35.430`).
- **Stabilitas spektral dan anggaran latih identik**: matriks hasil pengacakan memiliki *spectral radius* mentah `0,1302` lalu diskalakan kembali ke `0,9500`, dibekukan (*frozen*), dan dilatih dengan PPO pada `70.006` parameter adapter yang sama persis (`model_type = "fly_shuffled"`).

Dengan kontrol ini, hasil perbandingan terhadap `Fly Connectome PPO` memiliki interpretasi objektif apa pun keluarannya: keunggulan `Fly Connectome` mengindikasikan peran motif sirkuit biologis; performa setara menunjukkan dominasi keseimbangan E/I, *sparsity*, dan ekspansi dimensi `2.500`; sedangkan keunggulan `Shuffled` mengindikasikan bahwa spesialisasi struktural biologis justru membatasi pencampuran sinyal pada domain persimpangan.

---

## Bagian II: Lingkungan Simulasi Persimpangan

Lingkungan simulasi (`TrafficIntersectionEnv` di `src/fly_traffic/environment/traffic_env.py`) mengimplementasikan antarmuka Gymnasium berbasis NumPy yang berjalan sepenuhnya di CPU.

![Geometri 8 Lajur & 4 Fase Sinyal](intersection_2d_diagram.png)

![Alur Langkah Environment per Episode](alur-langkah-env.svg)

### Geometri Persimpangan dan Pengontrol Sinyal

Persimpangan memiliki 4 pendekatan (Utara, Selatan, Timur, Barat), masing-masing terdiri atas lajur lurus dan lajur belok kiri, sehingga total ada 8 lajur (`NUM_LANES = 8`). Tiap lajur dikelola sebagai antrean FIFO (`LaneQueue`) yang mencatat langkah kedatangan setiap kendaraan untuk menghitung waktu tunggu secara akurat.

Terdapat 4 fasa sinyal non-konflik (`SignalPhase`):

| Indeks Fasa | Lajur Hijau | Label |
|---|---|---|
| `0` | Utara Lurus, Selatan Lurus | `NS Straight` |
| `1` | Utara Belok, Selatan Belok | `NS Turn` |
| `2` | Timur Lurus, Barat Lurus | `EW Straight` |
| `3` | Timur Belok, Barat Belok | `EW Turn` |

Setiap pergantian fasa dikawal oleh mesin status `SignalController` agar lampu hijau tidak pernah langsung berpindah ke fasa yang berkonflik sebelum durasi aman terpenuhi:

```python
# File: src/fly_traffic/environment/intersection_state.py
# Metode: SignalController.request_phase() (baris 150-168)

def request_phase(self, target_phase: SignalPhase) -> bool:
    if self.is_transitioning:
        return False
    if target_phase == self.active_phase:
        return False
    if self.phase_elapsed < self.min_green_steps:
        return False

    self.pending_phase = target_phase
    if self.yellow_steps > 0:
        self.stage = SignalStage.YELLOW
        self.stage_elapsed = 0
    elif self.all_red_steps > 0:
        self.stage = SignalStage.ALL_RED
        self.stage_elapsed = 0
    else:
        self._activate_pending_phase()
    return True
```

Secara bawaan, `min_green_steps = 6` (`12,0 s`), `yellow_steps = 2` (`4,0 s`), dan `all_red_steps = 1` (`2,0 s`) dengan durasi satu langkah `dt_seconds = 2,0` detik.

### Ruang Observasi dan Ruang Aksi

Pada setiap langkah, agen menerima vektor observasi `22`-dimensi ($x_t$) dan memilih satu aksi diskrit dari `Discrete(4)` (fasa sinyal target):

- `obs[0:8]` — panjang antrean tiap lajur, dinormalisasi dengan `30,0` dan dibatasi ke `[0,0, 2,0]`.
- `obs[8:16]` — waktu tunggu kepala antrean tiap lajur (detik), dinormalisasi dengan `150,0` dan dibatasi ke `[0,0, 2,0]`.
- `obs[16:20]` — vektor *one-hot* fasa sinyal yang sedang aktif.
- `obs[20]` — penanda tahap transisi (`1,0` saat `YELLOW` atau `ALL_RED`, `0,0` saat `GREEN`).
- `obs[21]` — durasi fasa hijau berjalan, dinormalisasi dengan `20,0` dan dibatasi ke `[0,0, 2,0]`.

### Fungsi *Reward*

Setiap langkah simulasi menghasilkan skalar *reward* melalui metode `_compute_reward` (`traffic_env.py`):

$$r_t = \text{reward\_scale} \cdot \Big(-\sum_{i=0}^{7} q_i - \alpha \cdot w_{\max} - \beta \cdot \mathbb{I}(\text{switch}) - c \cdot n_{\text{col}}\Big)$$

dengan komponen bawaan:

- $\sum q_i$: jumlah total kendaraan mengantre di 8 lajur.
- $\alpha = 0,1$ (`wait_penalty_alpha`): penalti terhadap waktu tunggu maksimum $w_{\max}$ (detik).
- $\beta = 1,0$ (`switch_penalty_beta`): penalti setiap kali fasa sinyal berpindah agar agen tidak berganti lampu terlalu sering.
- $c = 80,0$ (`collision_penalty`): penalti per kejadian tabrakan $n_{\text{col}}$.
- $\text{reward\_scale} = 0,005$: faktor skala agar nilai *reward* tetap stabil selama optimasi PPO (satu kolisi menurunkan *reward* langkah sebesar `80,0 × 0,005 = 0,400`).

### Model Kolisi dan *Safety Buffer*

Ketika `enable_collisions = True`, `TrafficIntersectionEnv._evaluate_collisions` memodelkan dua jenis pelanggaran/konflik di persimpangan:

1. **Konflik Lampu Merah (`red_light_conflict`)** — terjadi ketika suatu lajur merah menunggu lebih lama dari `red_light_wait_threshold_s = 150,0` detik (*queue starvation*) sementara ada lajur hijau yang berkonflik sedang melepas kendaraan. Probabilitas pelanggaran meningkat linear hingga batas `red_light_max_prob = 0,14`:

   $$p_{\text{red}} = \min\left(0,14,\ \frac{\text{head\_wait\_s} - 150,0}{140,0} \cdot 0,14\right)$$

2. **Konflik Pengosongan Kotak Persimpangan (`box_clearance_conflict`)** — saat fasa berpindah (`phase_switched = True`), lajur yang baru saja melepas berpeluang `dilemma_zone_prob = 0,18` menyisakan kendaraan tertinggal (*straggler*) di tengah persimpangan dengan sisa umur `2` langkah.

```python
# File: src/fly_traffic/environment/traffic_env.py
# Potongan dari _evaluate_box_clearance_conflicts (baris 454-504)

if phase_switched:
    for lane_index, departed in enumerate(previous_departures):
        if departed > 0 and float(self.np_random.random()) < self.config.dilemma_zone_prob:
            self._box_stragglers.append([lane_index, 2])

for straggler_lane, remaining_steps in self._box_stragglers:
    if remaining_steps <= 0:
        continue
    conflicting_set = CONFLICTING_LANES[straggler_lane]
    conflicting_victims = [
        lane_index for lane_index in discharging_lanes
        if lane_index in conflicting_set
    ]
    if conflicting_victims:
        # Catat box_clearance_conflict dengan korban acak dari lajur berkonflik
        ...
```

Dengan pengaturan standar (`yellow_steps = 2` dan `all_red_steps = 1`), tersedia 3 langkah jeda tanpa pelepasan kendaraan sehingga umur *straggler* (`2 → 1 → 0`) habis sebelum fasa baru mendapat lampu hijau. Konflik `box_clearance_conflict` muncul apabila total jeda antar-hijau (`yellow_steps + all_red_steps`) dikurangi di bawah 2 langkah sehingga fasa baru menyala hijau selagi `remaining_steps > 0`.

---

## Bagian III: Arsitektur Pengendali (Agents)

Lima pengendali tersedia pada lingkungan dan metrik yang identik. Keempat model pembelajaran mesin disetel agar memiliki jumlah parameter terlatih yang setara (sekitar 70 ribu parameter):

| Pengendali | Kelas / Modul | Memori Rekuren | Parameter Terlatih |
|---|---|---|---|
| **Fly Connectome PPO** (`fly`) | `FlyConnectomePolicy` (`fly_policy.py`) | Ya (`2.500` neuron *sparse* FAFB v783 asli) | `70.006` |
| **Shuffled Connectome PPO** (`fly_shuffled`) | `FlyConnectomePolicy` + `shuffled.py` | Ya (`2.500` neuron *sparse* teracak, *null model*) | `70.006` |
| **PPO-GRU** (`gru`) | `GRUBaselinePolicy` (`baselines.py`) | Ya (`105` unit GRU) | `69.725` |
| **PPO-MLP** (`mlp`) | `MLPBaselinePolicy` (`baselines.py`) | Tidak (2 lapis *feedforward*, lebar `250`) | `69.755` |
| **Fixed Timer** (`fixed_timer`) | `FixedTimerAgent` (`fixed_timer.py`) | Tidak (siklus tetap `(10, 6, 10, 6)` langkah) | `0` |

### Paradigma Reservoir Biologis dan Integrasi Adapter

Secara arsitektural, proyek ini menerapkan pola **Biological Reservoir Computing**:
- **Inti Otak Lalat Dibekukan (*Frozen Connectome Core*):** `2.500` neuron (1.800 CX + 700 MB) beserta `87.171` bobot sinaps `W_exc` dan `W_inh` dimuat sebagai `register_buffer` (`requires_grad = False`). Struktur graf dan bobot internal otak lalat tidak dimodifikasi selama pelatihan dan tidak memiliki pemetaan manual satu-per-satu antara fitur lalu lintas dengan ID neuron tertentu.
- **Lapisan Adapter Terlatih (*Data-Driven Interfaces*):** Hanya `input_projection` (`nn.Linear(22, 2500)`, proyeksi dense ke seluruh 2.500 neuron), `tau_logit` (`1` skalar), `actor_head` (`2500 -> 4`), dan `critic_head` (`2500 -> 1`) yang diperbarui oleh Adam. Gradien dari PPO tetap mengalir mundur menembus $W_{\text{exc}}^T$ dan $W_{\text{inh}}^T$ (BPTT) sehingga optimizer otomatis menentukan kombinasi pembebanan tiap neuron tanpa campur tangan manual.

### Fly Connectome Policy

Inti kebijakan biologis adalah sel *leaky integrator* rekuren `ConnectomeRecurrentCell` yang memperbarui vektor aktivasi $h_t \in \mathbb{R}^{2500}$ di setiap langkah:

$$h_{t+1} = (1 - \tau) h_t + \tau \tanh\Big((g_{\text{exc}} W_{\text{exc}} - g_{\text{inh}} W_{\text{inh}}) h_t + W_{\text{in}} x_t + b_{\text{in}}\Big)$$

Keterangan komponennya:

| Komponen | Status | Detail |
|---|---|---|
| $W_{\text{exc}}$, $W_{\text{inh}}$ | **Frozen** (`requires_grad=False`) | `register_buffer`; 36.908 sisi eksitatori, 50.263 sisi inhibitori; gradien lewat $W^T$ saat BPTT |
| `input_projection` (`nn.Linear(22, 2500)`) | **Trainable** | 55.000 bobot + 2.500 bias = **57.500** parameter |
| `actor_head` (`2500 → 4`) | **Trainable** | 2.500 × 4 + 4 = **10.004** parameter |
| `critic_head` (`2500 → 1`) | **Trainable** | 2.500 × 1 + 1 = **2.501** parameter |
| `tau_logit` (skalar) | **Trainable** | 1 parameter; memetakan $\tau \in (0.05, 0.95)$ via sigmoid |
| **Total parameter terlatih** | | **70.006** |
| $g_{\text{exc}}$ (ACh gain) | Runtime control | Slider `[0.0, 2.5]` di Web UI |
| $g_{\text{inh}}$ (GABA/Glu gain) | Runtime control | Slider `[0.0, 2.5]` di Web UI |
| `lesion_mask` (biner) | Runtime control | Nonaktifkan subset neuron CX/MB/acak tanpa rebuild sparse matrix |

```python
# File: src/fly_traffic/agents/connectome_cell.py
# Metode: ConnectomeRecurrentCell.forward() (baris 107-127)

def forward(self, obs: torch.Tensor, hidden_state: torch.Tensor) -> torch.Tensor:
    mask = self.lesion_mask
    masked_hidden = hidden_state * mask

    hidden_t = masked_hidden.transpose(0, 1)
    exc_drive = torch.sparse.mm(self.w_exc_csr, hidden_t).transpose(0, 1)
    inh_drive = torch.sparse.mm(self.w_inh_csr, hidden_t).transpose(0, 1)

    synaptic_input = (
        self.g_exc * exc_drive
        - self.g_inh * inh_drive
        + self.input_projection(obs)
    )
    tau = self.leak_rate
    next_hidden = (1.0 - tau) * masked_hidden + tau * torch.tanh(synaptic_input)
    return next_hidden * mask
```

Pada `FlyConnectomePolicy.forward_step()` (`src/fly_traffic/agents/fly_policy.py`, baris 59–68), `next_hidden` diteruskan ke `actor_head` untuk menghasilkan *logits* 4 fasa dan ke `critic_head` untuk menghasilkan skalar $V(s_t)$.

---

## Bagian IV: Pelatihan PPO di CPU

Ketiga agen neural dilatih di CPU menggunakan *Proximal Policy Optimization* (`PPOTrainer` pada `src/fly_traffic/training/ppo_trainer.py`).

### Konfigurasi Rollout dan Masking Keputusan

- **Lingkungan Paralel**: 8 instans `TrafficIntersectionEnv` dijalankan bersamaan via `SyncVectorEnv` (`max_steps = 500` / `1.000` detik) dengan profil `mixed`. Tiap reset episode memilih secara acak terbobot salah satu dari `normal` (`~22,2%`), `asymmetric` (`~22,2%`), `asymmetric_ew` (`~22,2%`), `rush_hour` (`~22,2%`), atau `stress_train` (`~11,1%`), disertai pengacakan laju kedatangan `Uniform(0,85, 1,15)`.
- **Jadwal Training**: `rollout_steps = 128` menghasilkan `8 × 128 = 1.024` transisi per update. Untuk pelatihan standar `163.840` langkah, proses berjalan selama `160` update dengan *learning rate* menurun linear dari `8e-4` menjadi `1,2e-4` (`anneal_lr = True`).
- **Decision Mask**: saat sinyal berada pada tahap `YELLOW`, `ALL_RED`, atau lampu `GREEN` yang belum mencapai `min_green_steps = 6`, agen secara fisik tidak dapat mengganti fasa. `_compute_decision_mask()` menandai langkah-langkah terkunci tersebut bernilai `0` agar gradien aktor dan entropi hanya dihitung pada langkah pengambilan keputusan yang sah.

### Alur Rollout dan Pembaruan Bobot PPO

**1. Fase Pengumpulan Rollout (per Update `1..160`)**

```mermaid
flowchart LR
    A["Anneal LR<br/>lr = max(1e-6, decay * 8e-4)"] --> B["Reset hidden state<br/>jika episode sebelumnya selesai"]
    B --> C["Hitung decision_mask<br/>(GREEN & phase_elapsed >= 6)"]
    C --> D["policy.get_action_and_value<br/>(obs, hidden) -> action, value, next_hidden"]
    D --> E["vector_envs.step(action)<br/>terima next_obs, reward, done"]
    E --> F["Simpan ke RecurrentRolloutBuffer<br/>(ulangi 128 langkah x 8 env = 1.024)"]
    F --> G["Bootstrap V(s_128) & hitung GAE<br/>(gamma = 0.99, gae_lambda = 0.95)"]
```

**2. Fase Optimasi Sekuens (`_ppo_update`)**

```mermaid
flowchart LR
    H["Pecah buffer [128, 8]<br/>menjadi 64 chunk (len = 16)"] --> I["4 Epoch x 4 Minibatch acak<br/>(16 chunk per minibatch)"]
    I --> J["forward_sequence<br/>(reset hidden di dones_seq)"]
    J --> K["Normalisasi advantage &<br/>Policy Loss + Entropy Loss<br/>(hanya pada decision_mask == 1)"]
    K --> L["Value Loss MSE<br/>(pada semua langkah, vf_coef = 0.25)"]
    L --> M["Backprop + clip_grad_norm_(0.5)<br/>Adam update W_in, heads, tau_logit<br/>(W_exc & W_inh tetap frozen)"]
```

Di bawah ini adalah bagian inti dari `_ppo_update()` yang menghitung ketiga komponen kerugian dan memperbarui parameter terlatih:

```python
# File: src/fly_traffic/training/ppo_trainer.py
# Potongan dari PPOTrainer._ppo_update() (baris 361-425)

action_logits, predicted_values, _ = policy.forward_sequence(
    obs_seq=minibatch.obs_seq,
    initial_hidden=minibatch.initial_hidden,
    dones_seq=minibatch.episode_starts_seq,
)
action_distribution = Categorical(logits=action_logits)
new_log_probs = action_distribution.log_prob(minibatch.actions_seq)
step_entropy = action_distribution.entropy()

probability_ratio = torch.exp(new_log_probs - minibatch.old_log_probs_seq)
decision_mask = minibatch.decision_masks_seq
valid_decision_count = decision_mask.sum().clamp(min=1.0)

advantages = minibatch.advantages_seq
advantage_mean = (advantages * decision_mask).sum() / valid_decision_count
advantage_variance = (
    ((advantages - advantage_mean) ** 2 * decision_mask).sum()
    / valid_decision_count
)
normalized_advantages = (advantages - advantage_mean) / (advantage_variance.sqrt() + 1e-8)

unclipped_surrogate = -normalized_advantages * probability_ratio
clipped_surrogate = -normalized_advantages * torch.clamp(
    probability_ratio, 1.0 - ppo_config.clip_coef, 1.0 + ppo_config.clip_coef
)
policy_loss = (
    torch.max(unclipped_surrogate, clipped_surrogate) * decision_mask
).sum() / valid_decision_count

value_loss = 0.5 * ((predicted_values - minibatch.returns_seq) ** 2).mean()
mean_entropy = (step_entropy * decision_mask).sum() / valid_decision_count
total_loss = policy_loss - ppo_config.ent_coef * mean_entropy + ppo_config.vf_coef * value_loss

optimizer.zero_grad()
total_loss.backward()
nn.utils.clip_grad_norm_(
    (p for p in policy.parameters() if p.requires_grad),
    ppo_config.max_grad_norm,
)
optimizer.step()
```

---

## Bagian V: Antarmuka Web Lab & Visualisasi 3D Otak Lalat

Seluruh siklus inferensi, manipulasi farmakologis, pelatihan CPU, dan evaluasi dapat dijalankan lewat server web lokal (`TrafficLabServer` di `src/fly_traffic/web/server.py`) yang berkomunikasi dengan browser melalui REST API dan dua kanal WebSocket (`/ws/simulation` dan `/ws/training`):

- **Simulasi Persimpangan 3D & 2D (`Intersection3DRenderer` & `IntersectionRenderer`)**: tab utama menampilkan **3D View** Three.js (dengan sakelar instan ke **2D Map**) yang tersinkronisasi 1-ke-1 dengan geometri lajur kanan (RHT) dan `CONFLICTING_LANES` Python (`N_STR..W_TRN`). Dilengkapi 3 preset kamera (`Top View`, `3D Angle`, `Street`), `OrbitControls`, **kompas mata angin hitam-putih statis** di tengah simpang (`N, E, S, W, NE, SE, SW, NW`), HUD sinyal & pengatur waktu per pendekatan, badge antrean, injeksi kendaraan dengan klik langsung pada lajur, animasi antrean kinematik dari koridor belakang, lengkung belok kiri C1-kontinu, jarak ikuti (*car-following gap*), serta koreografi kolisi (`red_light_conflict` & `box_clearance_conflict`). Aplikasi 3D terpisah juga dapat dijalankan via `simulation3d/serve.py`.
- **Visualisasi 3D Otak *Drosophila* (`BrainAssetBuilder` & `Brain3DRenderer`)**: dibangun dari mesh permukaan seluruh otak FlyWire (`brain_mesh_v3`), 78 mesh neuropil JFRC2 dalam ruang FAFB14.1, dan koordinat soma serta tipe sel v783 (2.499 dari 2.500 neuron cocok dengan `root_id`). Di browser, Three.js menampilkan:
  - Lokasi 2.500 neuron dalam dua mode tata letak (**Neuropil** maupun **Soma**) dengan pewarnaan aktivasi *live* $h_t$ (`-0,40` biru inhibitori hingga `+0,40` oranye eksitatori) dan penanda cincin merah untuk neuron yang dilesikan.
  - 3.000 sinapsis terkuat dengan pulsa arah aliran sinyal dari neuron presinaptik ke postsinaptik.
  - Inspektor neuron individual saat diklik (`root_id`, `cell_type`, neuropil dominan, neurotransmiter, dan nilai $h_t$), serta fallback matriks 2D `50 × 50`.
- **Training Studio**: menjalankan `TrainingJobManager` di thread terpisah dengan dukungan arsitektur `fly`, `fly_shuffled`, `mlp`, dan `gru`, kustomisasi hyperparameter PPO, penyimpanan berkas *checkpoint* kustom (`.pt`), serta pratinjau deterministik 4 langkah (*Live Policy Preview*) pada skenario `rush_hour` setiap selesai satu update PPO.
- **Benchmark & Connectome (`BenchmarkInspectorView` & `BenchmarkPlotter`)**: menyediakan pemilih model interaktif untuk membandingkan kombinasi model bawaan, `Shuffled Connectome PPO`, maupun *checkpoint* kustom `.pt` melalui `POST /api/benchmark/run` (`agent_keys`), lengkap dengan grafik batang komparatif `mean_reward`, `mean_wait_time_s`, dan `mean_collisions` baik di antarmuka web maupun berkas gambar Matplotlib (`training_curves.png`, `scenario_comparison.png`, dan `pharmacology_ablation.png`).
- **Eksplorasi Arsitektur 3D (`ArchitectureView`)**: halaman khusus yang memodelkan topologi *end-to-end* `FlyConnectomePolicy` dalam ruang 3D (`x_t`, `input_projection`, `ConnectomeRecurrentCell` dengan 2.500 neuron FAFB + $\tau$ + kontrol runtime, `actor_head`, `critic_head`, dan umpan balik lingkungan). Dilengkapi sakelar aliran **Forward Pass** vs **Backward Gradient**, filter status (*Trainable*, *Frozen Bio*, *Runtime*), peta komponen horizontal, inspektor per modul, serta panduan matematis 4 tahap.

---

## Bagian VI: Evaluasi dan Hasil

Evaluasi dilakukan secara deterministik menggunakan `AgentEvaluator` (`src/fly_traffic/evaluation/evaluator.py`) pada empat skenario dengan **10 seed identik** (`1000` sampai `1009`), masing-masing berdurasi `500` langkah (`1.000` detik simulasi):

1. **`normal`** — kedatangan kendaraan seimbang ringan–sedang di keempat arah (`0,12` lurus, `0,06` belok per langkah).
2. **`asymmetric`** — koridor Utara–Selatan padat berkelanjutan (`0,34 / 0,14`), Timur–Barat ringan (`0,06 / 0,04`).
3. **`rush_hour`** — tiga fase berurutan: moderat (`0,22 / 0,10` pada langkah `0–165`), lonjakan Utara–Selatan (`0,56 / 0,24` vs `0,16 / 0,08` pada `166–332`), dan lonjakan Timur–Barat (`333–499`).
4. **`stress_test`** — kondisi beban ekstrem di mana seluruh lajur mengalami laju kedatangan tinggi simultan (`ns = 0,50/0,40`, `ew = 0,50/0,40` kend/langkah) untuk menguji saturasi antrean dan batas kesabaran lampu merah (`red_light_wait_threshold_s = 150,0` detik).

### Perbandingan Multi-Skenario (10 Seed, 500 Langkah / Episode)

Angka di bawah disinkronkan langsung dari `artifacts/benchmarks/benchmark_table.md` dan `artifacts/benchmarks/benchmark_summary.json`:

| Skenario | Agen | Mean Reward | Mean Queue | Mean Wait (s) | Mean Throughput | Mean Switches | Mean Collisions | Latency (ms/step) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `normal` | Fly Connectome PPO | -35.446 | 9.32 | 25.82 | 352.4 | 54.7 | 0.00 | 0.306 |
| `normal` | Shuffled Connectome PPO | -35.630 | 9.37 | 25.99 | 353.2 | 55.0 | 0.00 | 0.299 |
| `normal` | PPO-GRU | -34.923 | 9.32 | 26.07 | 348.0 | 53.5 | 0.00 | 0.115 |
| `normal` | PPO-MLP | -34.605 | 9.21 | 25.75 | 349.1 | 52.9 | 0.00 | 0.106 |
| `normal` | Fixed Timer (`10-6-10-6`) | -40.626 | 11.01 | 30.69 | 348.8 | 45.0 | 0.00 | 0.002 |
| `asymmetric` | Fly Connectome PPO | -48.370 | 13.75 | 23.57 | 571.5 | 55.0 | 0.00 | 0.336 |
| `asymmetric` | Shuffled Connectome PPO | -47.624 | 13.88 | 23.83 | 570.1 | 55.0 | 0.00 | 0.270 |
| `asymmetric` | PPO-GRU | -47.893 | 13.34 | 22.63 | 577.1 | 53.6 | 0.00 | 0.129 |
| `asymmetric` | PPO-MLP | -47.616 | 13.47 | 23.27 | 565.5 | 53.5 | 0.00 | 0.089 |
| `asymmetric` | Fixed Timer (`10-6-10-6`) | -61.537 | 19.12 | 32.92 | 569.5 | 45.0 | 0.00 | 0.002 |
| `rush_hour` | Fly Connectome PPO | -79.574 | 25.65 | 27.96 | 886.7 | 54.8 | 0.00 | 0.288 |
| `rush_hour` | Shuffled Connectome PPO | -79.431 | 25.60 | 28.05 | 879.7 | 54.3 | 0.00 | 0.275 |
| `rush_hour` | PPO-GRU | -80.937 | 26.01 | 28.09 | 894.5 | 53.9 | 0.00 | 0.107 |
| `rush_hour` | PPO-MLP | -76.871 | 24.71 | 27.18 | 879.3 | 54.0 | 0.00 | 0.090 |
| `rush_hour` | Fixed Timer (`10-6-10-6`) | -104.742 | 35.43 | 38.61 | 855.9 | 45.0 | 0.00 | 0.002 |
| `stress_test` | Fly Connectome PPO | -399.038 | 144.89 | 88.38 | 1448.8 | 54.9 | 5.00 | 0.293 |
| `stress_test` | Shuffled Connectome PPO | -395.579 | 140.52 | 84.97 | 1460.1 | 54.9 | 14.00 | 0.296 |
| `stress_test` | PPO-GRU | -361.209 | 129.12 | 78.82 | 1468.7 | 52.6 | 8.60 | 0.109 |
| `stress_test` | PPO-MLP | -381.887 | 131.83 | 78.69 | 1487.1 | 53.8 | 25.20 | 0.088 |
| `stress_test` | Fixed Timer (`10-6-10-6`) | -365.067 | 126.19 | 76.47 | 1482.8 | 45.0 | 20.90 | 0.002 |

Pada skenario `normal`, `asymmetric`, dan `rush_hour` dengan ambang batas kesabaran lampu merah `150,0` detik dan konfigurasi intergreen bawaan (`yellow_steps = 2`, `all_red_steps = 1`), seluruh agen pada `benchmark_summary.json` mencatat rata-rata `0,00` kolisi dan seluruh agen RL mengungguli `Fixed Timer` secara konsisten (`rush_hour`: `PPO-MLP -76,871 / 27,18 s`, `Shuffled -79,431 / 28,05 s`, `Fly -79,574 / 27,96 s`, `PPO-GRU -80,937 / 28,09 s` vs `Fixed Timer -104,742 / 38,61 s`). Pada profil beban ekstrem `stress_test`, `PPO-GRU` (`-361,209 +/- 22,370`, `52,6` switches, `8,60` kolisi) menjadi satu-satunya agen RL yang mengungguli `Fixed Timer` (`-365,067 +/- 17,968`, `20,90` kolisi) pada `Mean Reward`, sementara `Fly Connectome PPO` mencatat tingkat konflik terendah di antara seluruh agen (`5,00` kolisi vs `14,00` pada `Shuffled` dan `25,20` pada `PPO-MLP`) serta standar deviasi *reward* paling rapat (`+/- 17,601`).

### Visualisasi Hasil Benchmark

![Perbandingan Mean Reward Lintas Skenario](../artifacts/benchmarks/mean_reward.png)

![Perbandingan Mean Collisions Lintas Skenario](../artifacts/benchmarks/mean_collisions.png)

![Perbandingan Lintas Skema Pelatihan (Env v1 vs Stress Training vs Env v2)](../artifacts/benchmarks/experiment_evolution_comparison.png)

![Perbandingan 4 Metrik Skenario](../artifacts/benchmarks/scenario_comparison.png)

![Kurva Pelatihan PPO](../artifacts/benchmarks/training_curves.png)

> Arsip data komparasi lintas skema pelatihan (`Env v1`, `Stress Training`, dan konfigurasi standar `500` langkah saat ini) tersedia di [experiment_evolution.json](../artifacts/benchmarks/experiment_evolution.json).

### Uji Farmakologi dan Lesi Sirkuit (`rush_hour`, 10 Seed)

Pengukuran ketahanan `Fly Connectome PPO` ketika gain sinaptik atau sebagian neuron dimodifikasi (`artifacts/benchmarks/benchmark_summary.json`):

![Hasil Uji Farmakologi dan Lesi Sirkuit](../artifacts/benchmarks/pharmacology_ablation.png)

| Kondisi | `g_exc` | `g_inh` | Neuron Aktif | Mean Reward | Mean Wait (s) | Mean Throughput | Mean Switches | Mean Collisions |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline (`g_exc=1.0, g_inh=1.0`) | 1.00 | 1.00 | 2500 | -79.574 | 27.96 | 886.7 | 54.8 | 0.00 |
| High Excitation (`g_exc=1.8`) | 1.80 | 1.00 | 2500 | -78.480 | 27.44 | 885.2 | 54.4 | 0.00 |
| Low Inhibition (`g_inh=0.2`) | 1.00 | 0.20 | 2500 | -78.988 | 27.93 | 881.6 | 54.7 | 0.00 |
| High Inhibition (`g_inh=2.0`) | 1.00 | 2.00 | 2500 | -79.107 | 28.01 | 879.1 | 54.7 | 0.00 |
| CX Lesion 25% | 1.00 | 1.00 | 2050 | -80.407 | 28.18 | 886.2 | 54.6 | 0.00 |
| MB Lesion 25% | 1.00 | 1.00 | 2325 | -78.734 | 27.55 | 887.3 | 54.8 | 0.00 |
| Random Lesion 25% | 1.00 | 1.00 | 1875 | -79.440 | 27.94 | 885.2 | 54.6 | 0.00 |

### Mekanisme *Safety Buffer* (`All-Red` vs `Yellow Only`)

Kontrol `set_safety_buffer()` (dan sakelar UI **Signal Buffer**) mengatur parameter `all_red_steps` (`1` = `2,0 s` jeda seluruh arah merah setelah `yellow_steps = 2`, atau `0` = *Yellow Only*). Karena umur *straggler* di `_box_stragglers` bernilai `2` langkah, konflik pengosongan kotak simpang (`box_clearance_conflict`) tertekan ke `0` selama total jeda antar-hijau berjumlah `>= 2` langkah, dan langsung meningkat bila durasi transisi diturunkan di bawah `2` langkah.

---

## Bagian VII: Batasan Kondisi dan *Trade-off*

1. **Abstraksi simulasi terbatas.** `TrafficIntersectionEnv` adalah model antrean diskrit berbasis NumPy (ditampilkan secara visual lewat 3D Three.js dan 2D Canvas) tanpa fisika mikroskopik kendaraan seperti pada simulator SUMO. Hasilnya tidak dapat langsung digeneralisasikan ke persimpangan dunia nyata.
2. **Ukuran sampel evaluasi kecil.** Benchmark mengevaluasi 10 seed per skenario, sehingga perbedaan tipis antar arsitektur atau *checkpoint* belum cukup kuat untuk menyimpulkan signifikansi statistik tanpa pengujian seed yang lebih besar.
3. **Model kolisi bersifat stokastik.** `red_light_conflict` dan `box_clearance_conflict` menggunakan pencuplikan acak dengan probabilitas tetap. Variasi seed dapat menghasilkan jumlah konflik yang berbeda meski bobot kebijakannya sama.
4. **Biaya latensi operasi matriks *sparse*.** Pada CPU, inferensi satu langkah `Fly Connectome PPO` membutuhkan `0,288–0,336 ms/step` dan `Shuffled` `0,270–0,299 ms/step`, sekitar 2,5–3 kali lebih lambat dibanding `PPO-GRU` (`0,107–0,129 ms/step`) maupun `PPO-MLP` (`0,088–0,106 ms/step`), serta jauh di bawah `Fixed Timer` (`0,002 ms/step`).
5. **Keselamatan tetap bergantung pada *interlock* sinyal dan kapasitas simpang.** `SignalController` mengunci perpindahan fase minimum hijau `6` langkah serta fase transisi antar-hijau, namun kondisi jenuh di atas `red_light_wait_threshold_s = 150,0 s` tetap menimbulkan pelanggaran lampu merah pada kebijakan mana pun.

---

## Penutup

Berdasarkan pengukuran di workspace ini, `PPO-MLP` unggul pada kecepatan dan skenario non-jenuh (`normal` & `rush_hour`) namun memicu konflik terbanyak saat jenuh (`25,20`); `PPO-GRU` menjadi agen *end-to-end* terbaik hingga melampaui `Fixed Timer` di `stress_test` (`-361,209` vs `-365,067`); sementara sub-konektom *Drosophila melanogaster* (`Fly Connectome PPO`, `2.500` neuron CX dan MB *frozen*) mencatat tingkat keselamatan tertinggi (`5,00` kolisi vs `14,00` pada `Shuffled`) dan varians paling rapat sambil mempertahankan kendali *in-silico* atas penguatan sinaptik dan lesi neuron secara langsung. Kinerja tersebut diimbangi oleh latensi CPU lebih tinggi akibat operasi *sparse*, kekakuan *frozen reservoir* pada penahanan *switch*, serta keterbatasan lingkungan uji 2D yang bersifat abstrak.

---

## Lampiran: Glosari Istilah

### Neurosains & Konektom
- *Cross-region edges*: sisi sinaptik yang menghubungkan neuron di Central Complex dengan neuron di Mushroom Body.
- *Frozen / trainable parameters*: parameter yang dikunci selama training (matriks $W_{\text{exc}}, W_{\text{inh}}$) versus parameter yang diperbarui optimizer ($W_{\text{in}}, b_{\text{in}}$, kepala aktor/kritik, dan $\tau$).
- *Global random edge rewiring* (*Shuffled null model*): pengacakan pasangan `(post, pre)` secara seragam pada matriks sinapsis *sparse* sembari mempertahankan jumlah sisi, distribusi bobot E/I, *sparsity*, dan *spectral radius* guna mengisolasi kontribusi topologi biologis.
- *In-silico ablation / lesion mask*: penonaktifan sementara sebagian neuron (menetapkan aktivasi nol melalui masker biner) untuk menguji ketahanan sirkuit tanpa menghitung ulang graf.
- *Leak rate* ($\tau$): koefisien integrasi membran pada rentang `(0,05, 0,95)` yang menentukan porsi memori lama yang dipertahankan terhadap masukan baru.
- *Log1p weighting*: transformasi $\log(1 + \text{syn\_count})$ untuk memadatkan rentang jumlah sinaps sebelum normalisasi.
- *Sparse CSR*: format penyimpanan matriks jarang terkompresi per baris (*Compressed Sparse Row*) yang hanya menyimpan elemen non-nol.
- *Spectral radius*: magnitudo nilai eigen terbesar suatu matriks; diskalakan ke `0,95` untuk mencegah aktivasi rekuren meledak.

### *Reinforcement Learning* & PPO
- *Actor-critic*: arsitektur dengan dua keluaran dari representasi yang sama—kepala aktor (*logits* aksi) dan kepala kritik (estimasi nilai keadaan $V(s_t)$).
- *Decision mask*: masker biner yang memastikan gradien aktor dan entropi hanya dihitung pada langkah di mana perpindahan fasa diizinkan.
- *Generalized Advantage Estimation* (GAE): metode estimasi keuntungan (*advantage*) dengan faktor diskonto $\gamma$ dan parameter penyeimbang bias-varians $\lambda_{\text{GAE}}$.
- *Gradient clipping*: pembatasan norma gradien (maksimal `0,5`) sebelum pembaruan bobot untuk menjaga stabilitas optimasi.
- *PPO (Proximal Policy Optimization)*: algoritma pembelajaran penguat yang membatasi rasio perubahan kebijakan per pembaruan lewat *clipped surrogate objective*.
- *Sequence chunk*: potongan urutan rollout berdurasi `16` langkah untuk melatih jaringan rekuren per *minibatch*.

### Simulasi Lalu Lintas & Sistem
- *All-red steps*: langkah jeda saat seluruh arah menyala merah di antara tahap kuning dan fase hijau berikutnya guna mengosongkan kotak persimpangan.
- *Dilemma-zone straggler*: kendaraan yang berpeluang tertinggal di tengah persimpangan ketika fasa berganti (`box_clearance_conflict`).
- *Latency*: waktu komputasi satu langkah inferensi model pada CPU (milidetik per langkah).
- *Queue starvation*: kondisi ketika lajur yang terus mendapat lampu merah melebihi ambang tunggu `150,0` detik sehingga memicu peluang pelanggaran (`red_light_conflict`).
- *Throughput*: jumlah total kendaraan yang berhasil melewati persimpangan dalam satu episode.

---

*Dokumentasi teknis ini disusun berdasarkan implementasi dan hasil pengukuran aktual pada repositori proyek.*
