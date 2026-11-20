# Blueprint & Build Plan: Drosophila Connectome RL Traffic Controller

## 1. Konsep & Tujuan Proyek

Membangun agen *Reinforcement Learning* (RL) pengatur lampu lalu lintas persimpangan 4-arah yang menjadikan **sub-graf biologis otak lalat buah (`FAFB v783`)** sebagai inti rekuren (*recurrent policy core*) menggantikan *hidden layer* MLP/LSTM konvensional.

Fokus utamanya adalah menguji apakah topologi kabel evolusioner otak lalat—terutama mekanisme inhibisi silang (*Winner-Take-All*) dan memori spasial—dapat berfungsi sebagai *inductive bias* yang efektif untuk regulasi lalu lintas, sekaligus memungkinkan eksperimen manipulasi neurotransmitter secara *real-time*.

---

## 2. Batasan Hardware & Skala (CPU-Only, Non-Minimal)

* **Target Eksekusi:** 100% **CPU-Only** (lokal, tanpa GPU server).
* **Skala Otak:** **1.500–3.000 neuron** (bukan *ultra-minimal* puluhan neuron). Memanfaatkan struktur matriks *sparse* alami (>98% elemen nol) sehingga *forward pass* tetap cepat (~1–2 ms per *step* di CPU).
* **Skala Simulasi:** *Custom environment* 2D berbasis NumPy (tanpa simulator berat seperti SUMO/3D), namun tetap non-trivial: multi-lajur, transisi lampu kuning/*all-red*, dinamika antrean, dan lonjakan kepadatan asimetris (*rush hour*).

---

## 3. Sumber Data

Dari **FlyWire Codex (`codex.flywire.ai`)** dataset **`FAFB v783`**:

1. **`Connections (Filtered) (68 MB)`** — tabel *edge* sinaptik (`pre_root_id`, `post_root_id`, `syn_count`, `nt_type`, `neuropil`).


2. **Tabel Klasifikasi/Anotasi Neuron** (`Classification` / `Neurons`) — pemetaan `root_id` ke kelompok sirkuit otak.

---

## 4. Spesifikasi Modul yang Akan Dibangun

### Modul 1: Connectome Preprocessor (`Polars` + `SciPy/PyTorch Sparse`)

* Saring neuron pada sirkuit **Central Complex** (seleksi aksi & navigasi arah) dan sebagian **Mushroom Body** (asosiasi *reward*) hingga ukuran target 1.500–3.000 neuron.
* Normalisasi bobot sinapsis dengan $\log(1 + \text{syn\_count})$ lalu skalakan *spectral radius* agar dinamika rekuren stabil (tidak meledak/lenyap).
* Pecah menjadi dua *sparse adjacency matrix* terpisah sesuai polaritas biologis lalat:
* $W_{\text{exc}}$: koneksi bertipe *Acetylcholine* (eksitatori).
* $W_{\text{inh}}$: koneksi bertipe *GABA* dan *Glutamate* (inhibitori dominan di sirkuit pusat *Drosophila*).


* Simpan sebagai artefak praproses (`.pt` / `.npz`) agar tahap *training* tidak membaca ulang CSV.

### Modul 2: Traffic Environment (`Gymnasium` API, Pure Python/NumPy)

* **Geometri:** Persimpangan 4 arah (Utara, Selatan, Timur, Barat), masing-masing 2 lajur (lurus & belok kanan/kiri) dengan kedatangan kendaraan berbasis distribusi *Poisson* yang berubah profilnya (lancar, macet satu arah, *rush hour*).
* **Observation Space ($x_t$):** Panjang antrean per lajur, waktu tunggu maksimum mobil terdepan per jalur, *one-hot* fase lampu aktif, dan durasi fase berjalan.
* **Action Space ($a_t$):** Diskrit — pertahankan fase saat ini atau pindah ke fase berikutnya (termasuk *hardcoded yellow/all-red buffer* di level *environment* untuk mencegah tabrakan).
* **Reward Function ($r_t$):**

$$r_t = -\sum \text{queue\_length} - \alpha \cdot \text{max\_wait\_time} - \beta \cdot \mathbb{I}(\text{phase\_switch})$$



### Modul 3: Connectome RL Policy & Training (`PyTorch` + PPO)

* **Arsitektur Inti (*Custom Sparse Recurrent Cell*):**

$$h_{t+1} = (1 - \tau) h_t + \tau \tanh\Big((g_{\text{exc}} W_{\text{exc}} - g_{\text{inh}} W_{\text{inh}}) h_t + W_{\text{in}} x_t\Big)$$


* **Parameter Terlatih:** $W_{\text{exc}}$ dan $W_{\text{inh}}$ **dikunci (*frozen*)** untuk mempertahankan struktur biologis asli. RL hanya melatih proyeksi input ($W_{\text{in}}$), *policy/value heads* ($W_{\text{out}}$), dan opsional konstanta *leak rate* ($\tau$).
* **Training Loop:** PPO (menggunakan CleanRL atau Stable-Baselines3) dengan 4–8 *vectorized CPU environments*.
* **Baseline Pembanding:** Implementasikan agen *Fixed Timer* (statis) dan agen PPO-MLP/GRU standar dengan jumlah parameter terlatih setara.

### Modul 4: Visualisasi & Lab Eksperimen (`Pygame`)

* **Panel Kiri:** Visualisasi 2D persimpangan *real-time* (animasi antrean kendaraan, warna lampu, metrik *throughput* & rata-rata waktu tunggu).
* **Panel Kanan:** *Heatmap* aktivasi neuron lalat secara *live* selama mengambil keputusan.
* **Kontrol Interaktif (*In-Silico Pharmacology & Ablation*):** *Slider* pengali $g_{\text{exc}}$ (eksitasi) dan $g_{\text{inh}}$ (inhibisi/GABA) serta tombol *lesioning* (mematikan sebagian neuron) secara *real-time* untuk mengamati perubahan perilaku kontrol lampu.

---

## 5. Tahapan Eksekusi (Build Roadmap)

1. **Fase 1 — Ekstraksi Graf:** Buat skrip filter dataset `FAFB v783`, verifikasi jumlah neuron, kepadatan *sparse*, rasio eksitatori/inhibitori, dan stabilitas *spectral radius*.


2. **Fase 2 — Environment & Baseline:** Bangun *Gymnasium environment* lalu validasi menggunakan kontroler *Fixed Timer* dan latih *baseline* PPO-MLP/GRU.
3. **Fase 3 — Integrasi Otak Lalat:** Implementasikan modul PyTorch *sparse recurrent core* dan latih dengan algoritma PPO yang sama pada CPU.
4. **Fase 4 — Evaluasi Komparatif:** Uji semua agen pada skenario kepadatan normal hingga *rush hour* ekstrem; catat *sample efficiency*, rata-rata waktu tunggu, dan kestabilan pergantian lampu.
5. **Fase 5 — Sandbox Interaktif:** Bangun antarmuka Pygame 60 FPS yang menghubungkan *checkpoint* model terlatih dengan *heatmap* neuron dan *slider* manipulasi neurotransmitter.

---

## 6. Output yang Diharapkan (Deliverables)

* **Artefak Sub-Connectome (`fly_subconnectome.pt`):** Matriks *sparse* $W_{\text{exc}}, W_{\text{inh}}$ bersih beserta indeks dan label tipe neuronnya.
* **Trained Checkpoints:** Model terlatih `fly_ppo.pt` dan `baseline_ppo.pt` yang dapat dieksekusi di CPU.
* **Hasil Benchmark Kuantitatif:** Grafik dan tabel perbandingan *Fixed Timer* vs *MLP/GRU* vs *Fly Connectome* (kurva *reward*, rata-rata waktu tunggu kendaraan, dan *throughput*).
* **Aplikasi Demo Interaktif:** Simulasi 2D *real-time* yang menampilkan performa pengaturan lampu oleh otak lalat sekaligus efek visual saat kadar neurotransmitter atau sirkuit sarafnya dimodifikasi langsung.