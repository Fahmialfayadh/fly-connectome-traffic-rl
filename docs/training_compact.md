# Arsitektur & Alur Sistem Simulasi Lalu Lintas (Versi Ringkas)

Dokumen ini merangkum desain inti sistem **Fly Traffic**—mencakup lingkungan simulasi, mekanisme *safety interlock*, desain profil lalu lintas, serta integrasi *training* dengan *Live Policy Preview*. Algoritma pembaruan bobot standar (PPO) diperlakukan sebagai mesin optimasi *black-box* sehingga fokus tetap pada komponen yang dirancang khusus dalam proyek ini.

---

## 1. Alur Sistem Keseluruhan (*High-Level Architecture*)

Diagram ini menunjukkan bagaimana skenario lalu lintas, agen kontroler, mesin *training*, dan antarmuka Web UI saling terhubung.

```mermaid
flowchart LR
    subgraph EnvDesign ["Desain Lingkungan & Skenario"]
        P1["Profil Training (Mixed + Jitter)<br/>Normal, Asymmetric, Rush Hour, Stress"]
        P2["Profil Benchmark (Deterministik)<br/>Evaluasi 10 Seed"]
        ENV["TrafficIntersectionEnv<br/>8 Lajur FIFO, 4 Fase, Safety Interlock"]
        P1 --> ENV
        P2 -.-> ENV
    end

    subgraph AgentDesign ["Desain Agen & Optimasi"]
        AG["Policy Network<br/>Fly Connectome (Sparse 2.500 Neuron)<br/>vs Baseline (MLP / GRU / Fixed Timer)"]
        MASK["Decision Masking<br/>Filter langkah valid di luar masa interlock"]
        PPO["PPO Training Engine<br/>(Standard Optimizer)"]
        ENV -->|"Observasi 22-dim & Reward"| AG
        AG -->|"Aksi: Pilih 1 dari 4 Fase"| ENV
        ENV --> MASK
        MASK --> PPO
        PPO -->|"Update Bobot Policy"| AG
    end

    subgraph WebUI ["Web UI & Visualisasi"]
        PREV["Live Policy Preview<br/>(4 Tick Rush Hour per Update)"]
        LIVE["Live Simulation & Benchmark<br/>(Auto-load Checkpoint)"]
        AG -.->|"Snapshot Bobot"| PREV
        PPO -.->|"Simpan Checkpoint (.pt)"| LIVE
    end
```

---

## 2. Desain Siklus 1 Langkah Simulasi (`TrafficIntersectionEnv`)

Setiap episode berjalan selama `500` langkah (`1.000` detik simulasi). Flowchart berikut merangkum urutan eksekusi aktual setiap langkah di dalam `TrafficIntersectionEnv`:

![Flowchart Langkah Environment per Episode](alur-langkah-env.svg)

---

## 3. Alur *Headless Training* & *Live Policy Preview*

Agar *training* di CPU tetap cepat tanpa terhambat *rendering* visual 60 FPS di browser, sistem memisahkan jalur komputasi *training* dari jalur visualisasi menggunakan mekanisme *preview* ringan.

```mermaid
flowchart TD
    START["Mulai Training dari Web UI"] --> VEC["Jalankan 8 Env Paralel (Headless di CPU)<br/>Tanpa rendering grafis & tanpa jeda real-time"]
    VEC --> OPT["Kumpulkan Rollout & Jalankan 1 Siklus Update PPO"]
    OPT --> HOOK["Callback Live Preview (~10 ms)<br/>Jalankan 4 langkah deterministik pada env terpisah (rush_hour)"]
    HOOK --> WS["Kirim Metrik & 4 Frame Preview via WebSocket"]
    WS --> UI["Web UI Memutar Preview di Canvas<br/>(Interval 260 ms/frame tanpa memblokir training)"]

    OPT --> DONE{"Target Timesteps Tercapai?"}
    DONE -- "Belum" --> VEC
    DONE -- "Ya" --> SAVE["Simpan Checkpoint & Auto-Reload ke Tab Live Simulation"]
```

---

## Ringkasan Poin Desain Utama

| Komponen yang Didesain | Fungsi Utama |
|---|---|
| **Profil Lalu Lintas `mixed`** | Mengacak skenario dan menambahkan *jitter* laju kedatangan agar agen tidak *overfit* pada satu pola kepadatan. |
| **Signal Safety Interlock** | Menegakkan durasi hijau minimum (`6` langkah) serta jeda `YELLOW` dan `ALL_RED` untuk mencegah konflik *box clearance*. |
| **Decision Masking** | Memastikan pembaruan *policy* hanya dihitung pada langkah di mana agen benar-benar diizinkan mengambil keputusan perpindahan fase. |
| **Decoupled Live Preview** | Menjalankan *training* secara *headless* di CPU sambil tetap menampilkan perkembangan perilaku agen secara *live* di browser dengan *overhead* minimal. |