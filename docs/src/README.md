# Referensi Implementasi Backend (`docs/src/`)

Dokumen di direktori ini merinci variabel internal, struktur tensor/state, rumus, serta alur eksekusi di dalam kode `src/fly_traffic/`.

| Dokumen | Fokus Teknis |
|---|---|
| [`docs/config.md`](../config.md) | Konstanta global dan 5 dataclass konfigurasi (`ConnectomeConfig`, `TrafficEnvConfig`, `PPOConfig`, `BenchmarkConfig`, `SandboxUIConfig`). |
| [`connectome.md`](connectome.md) | Pipeline ekstraksi sub-graf FAFB, skema `ConnectomeArtifact`, pembobotan `log1p`, normalisasi spectral radius, *shuffled rewiring*, dan koordinat 3D otak. |
| [`environment.md`](environment.md) | Mesin state `LaneQueue` & `SignalController`, matriks `CONFLICTING_LANES`, vektor `obs[22]`, fungsi *reward*, fisika kolisi, dan profil Poisson. |
| [`agents.md`](agents.md) | Persamaan `ConnectomeRecurrentCell`, inisialisasi parameter `FlyConnectomePolicy` / `MLP` / `GRU`, kontrol `g_exc`/`g_inh`/lesi, dan format checkpoint `.pt`. |
| [`training.md`](training.md) | Struktur `RecurrentRolloutBuffer`, komputasi GAE-Lambda, *sequence chunking*, `_compute_decision_mask`, dan rumus loss `_ppo_update()`. |
| [`evaluation.md`](evaluation.md) | Metrik `EpisodeMetrics` & `ScenarioBenchmarkResult`, protokol 7 kondisi farmakologi/lesi, profiling latensi CPU, dan API `BenchmarkPlotter`. |
| [`web_ui.md`](web_ui.md) | Skema JSON payload WebSocket (`/ws/simulation` & `/ws/training`), kuantisasi aktivasi base64, REST API, dan loop Pygame sandbox. |
