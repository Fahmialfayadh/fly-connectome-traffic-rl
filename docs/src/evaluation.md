# Evaluasi Deterministik & Visualisasi Benchmark (`src/fly_traffic/evaluation/`)

## 1. Struktur Data Evaluasi (`evaluator.py`)

| Dataclass | Lingkup | Field Utama |
|---|---|---|
| `EpisodeMetrics` | 1 episode (1 seed, 1 agen, 1 skenario) | `total_reward`, `mean_queue`, `mean_wait_time_s`, `max_wait_time_s`, `total_throughput`, `total_switches`, `total_collisions` |
| `ScenarioBenchmarkResult` | Agregasi 10 seed (`1000..1009`) | `mean_reward`, `std_reward`, `mean_queue`, `mean_wait_time_s`, `std_wait_time_s`, `mean_throughput`, `std_throughput`, `mean_switches`, `inference_ms_per_step`, `mean_collisions` |
| `PharmacologyResult` | Uji perturbasi `FlyConnectomePolicy` (`rush_hour`) | `condition`, `g_exc`, `g_inh`, `lesion_target`, `lesion_fraction`, `active_neurons`, `mean_reward`, `mean_wait_time_s`, `mean_throughput`, `mean_switches`, `mean_collisions` |

---

## 2. Protokol Pengujian (`AgentEvaluator`)

1. **Multi-Skenario Deterministik (`evaluate_all_agents`)**:
   - Menjalankan seluruh agen pada `("normal", "asymmetric", "rush_hour", "stress_test")` dengan 10 seed (`1000..1009`) menggunakan pemilihan aksi deterministik (`deterministic=True`, `argmax` logits).
2. **Profiling Latensi CPU (`measure_inference_latency_ms`)**:
   - Melakukan pemanasan `20` langkah (`latency_warmup_steps`) pada profil `normal`, lalu mencatat rata-rata waktu eksekusi `agent.select_action(obs, deterministic=True)` selama `200` langkah (`latency_measure_steps`) via `time.perf_counter()`, dilaporkan dalam satuan `ms/step`.
3. **Suite Farmakologi & Lesi (`evaluate_pharmacology_and_lesions`)**:
   - Mengevaluasi 7 kondisi pada `rush_hour` (`seed=42` untuk pemilihan indeks lesi), lalu mengembalikan `g_exc=1.0`, `g_inh=1.0`, dan `reset_lesion()` di blok `finally`:
     - `Baseline (g_exc=1.0, g_inh=1.0)`
     - `High Excitation (g_exc=1.8, g_inh=1.0)`
     - `Low Inhibition (g_exc=1.0, g_inh=0.2)`
     - `High Inhibition (g_exc=1.0, g_inh=2.0)`
     - `CX Lesion 25%` (`apply_region_lesion(..., "CX", 0.25, seed=42)`)
     - `MB Lesion 25%` (`apply_region_lesion(..., "MB", 0.25, seed=42)`)
     - `Random Lesion 25%` (`apply_random_lesion(0.25, seed=42)`)
4. **Keluaran Laporan (`save_reports`)**: menulis `benchmark_summary.json` dan `benchmark_table.md`.

---

## 3. Generator Grafik Matplotlib (`BenchmarkPlotter`)

| Metode | Berkas Default | Tata Letak & Metrik |
|---|---|---|
| `plot_training_curves(histories, output_path)` | `training_curves.png` | 1×3 subplot garis: `mean_episode_reward`, `mean_throughput`, dan `mean_collisions` terhadap `timesteps`. |
| `plot_scenario_comparison(results, output_path)` | `scenario_comparison.png` | 2×2 *grouped bar* (`mean_reward` ± std, `mean_wait_time_s` ± std, `mean_throughput` ± std, `mean_collisions`). |
| `plot_single_scenario_metric(results, ...)` | `mean_reward.png`, `mean_collisions.png` | *Grouped bar* tunggal per metrik lintas skenario untuk ringkasan laporan. |
| `plot_reward_vs_collisions(results, output_path)` | `reward_vs_collisions.png` | 1×2 *grouped bar* fokus pada `mean_reward` dan `mean_collisions` (dengan label angka di atas bar non-nol). |
| `plot_pharmacology_ablation(results, output_path)` | `pharmacology_ablation.png` | 1×3 *horizontal bar* kondisi farmakologi/lesi untuk `mean_reward`, `mean_wait_time_s`, dan `mean_collisions`. |
