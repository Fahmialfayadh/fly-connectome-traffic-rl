# Backend Web Lab & Pygame Sandbox (`src/fly_traffic/web/`, `src/fly_traffic/ui/`)

## 1. Rute & API Server (`TrafficLabServer`)

Server `aiohttp` merakit halaman SPA dengan memperluas marker `<!-- @include partials/... -->` di `frontend/index.html` (`SPA_ROUTES`: `/`, `/simulation`, `/training`, `/benchmark`, `/architecture`, `/docs`).

| Endpoint | Method | Parameter / Body JSON | Perilaku |
|---|---|---|---|
| `/api/overview` | `GET` | - | Mengembalikan metadata konektom, inventaris agen & checkpoint, state training, dan isi `benchmark_summary.json`. |
| `/api/simulation/control` | `POST` | `{"action": ..., ...}` | Menjalankan perintah ke `SimulationService`: `toggle_run`, `set_running`, `step_once`, `reset`, `set_speed`, `set_scenario`, `set_safety_buffer`, `set_continuous_mode`, `spawn_vehicles`, `spawn_random`, `spawn_surge`, `set_agent`, `set_pharmacology`, `apply_lesion`, `reset_brain`, `reload_checkpoint`. |
| `/api/training/start` | `POST` | `model_type`, `total_timesteps`, `learning_rate`, `save_mode`, `checkpoint_name`, `resume_checkpoint`, opsi hyperparameter PPO | Memulai `TrainingJobManager.start_job()` pada thread terpisah. |
| `/api/training/stop` | `POST` | - | Mengirim `cancel_event` ke job training aktif (opsional menyimpan checkpoint parsial bila `save_on_cancel=True`). |
| `/api/benchmark/run` | `POST` | `{"agent_keys": [...]}` | Menjalankan `AgentEvaluator` + `BenchmarkPlotter` via `asyncio.to_thread` tanpa memblokir event loop. |
| `/ws/simulation` | `WS` | - | Menyiarkan frame `simulation_frame` berkala sesuai `steps_per_second`. |
| `/ws/training` | `WS` | - | Menyiarkan event `training_update`, `training_completed`, `training_cancelled`, `training_error`. |

---

## 2. Telemetri Frame & Kuantisasi Aktivasi (`SimulationService`)

Untuk menjaga payload WebSocket ringkas saat memvisualisasikan `2.500` neuron pada frekuensi tinggi, aktivasi $h_t$ pada rentang `[-0.4, +0.4]` dan biner `lesion_mask` dikuantisasi ke byte `uint8` base64:

```python
scaled_u8 = np.clip(
    np.round(((hidden_np / 0.4) * 0.5 + 0.5) * 255.0),
    0.0,
    255.0,
).astype(np.uint8)
lesion_u8 = (lesion_mask_np > 0.5).astype(np.uint8)
```

| Kunci Payload `simulation_frame` | Isi |
|---|---|
| `snapshot` | Status 8 antrean, daftar kendaraan per lajur, status sinyal (`GREEN`/`YELLOW`/`ALL_RED`), kejadian kolisi aktif, throughput, waktu tunggu. |
| `policy_telemetry` | `action_probabilities` `[P0..P3]`, `chosen_action`, `value_estimate`. |
| `connectome_telemetry` | `is_fly_active`, `g_exc`, `g_inh`, `leak_rate_tau`, `active_neurons`, `cx_mean_abs_activity`, `mb_mean_abs_activity`, `activations_b64`, `lesion_mask_b64`. |
| `model_telemetry` | Telemetri adaptif per jenis pengendali aktif (`fly`, `fly_shuffled`, `gru`, `mlp`, `fixed_timer`). |

Di mode **Live Simulation**, `SimulationService` memanggil `set_lane_capacity_limit(hard_max_lane_capacity)` (`80` kendaraan/lajur) agar injeksi kendaraan manual pengguna tetap tertampung tanpa mengubah konstanta normalisasi observasi `30.0`.

---

## 3. *Headless Training* + *Live Policy Preview* (`TrainingJobManager`)

- Pada setiap selesai 1 update PPO (`_on_policy_update`), callback `_rollout_preview_frames()` menjalankan **4 langkah deterministik** (`torch.argmax(logits, dim=-1)`) pada satu `preview_env` terpisah berprofil `rush_hour` dan menyisipkan hasilnya ke field `preview_frames` dalam event `training_update`.
- Ketika training selesai (atau dibatalkan dengan `save_on_cancel=True`), `TrainingJobManager` otomatis mendaftarkan checkpoint baru melalui `self._simulation_service.register_checkpoint(..., activate=True)` agar langsung tersedia di tab Live Simulation.

---

## 4. Pygame Desktop Sandbox (`TrafficConnectomeSandboxApp`)

Menggunakan konfigurasi `SandboxUIConfig` (`1280 × 760` px, target `60 FPS`, panel kiri `680 px` untuk persimpangan 2D + panel kanan `600 px` untuk heatmap konektom `50 × 50` beserta kontrol slider `g_exc`/`g_inh` dan tombol lesi `25%`). Mendukung eksekusi tanpa display melalui `render_headless_screenshot(output_path, steps=45)`.
