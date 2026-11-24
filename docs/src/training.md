# Pelatihan PPO Vektorisasi (`src/fly_traffic/training/`)

## 1. Penyimpanan Rollout & GAE (`RecurrentRolloutBuffer`)

Buffer mengalokasikan tensor `[rollout_steps, num_envs, ...]` (default `[128, 8]` = `1.024` transisi per update) pada CPU:

| Tensor Buffer | Bentuk | Tipe | Keterangan |
|---|---|---|---|
| `observations` | `[128, 8, 22]` | `float32` | Vektor observasi sebelum langkah dieksekusi. |
| `actions` | `[128, 8]` | `int64` | Fase target (`0..3`) yang dipilih kebijakan. |
| `log_probs` | `[128, 8]` | `float32` | Log-probabilitas aksi lama $\log \pi_{\text{old}}(a_t \mid s_t)$. |
| `rewards` | `[128, 8]` | `float32` | Skalar reward dari lingkungan. |
| `episode_starts` | `[128, 8]` | `float32` | `1.0` pada langkah pertama episode baru (`previous_done_mask`). |
| `step_dones` | `[128, 8]` | `float32` | `1.0` saat episode berakhir (`terminated or truncated`). |
| `decision_masks` | `[128, 8]` | `float32` | `1.0` hanya jika sinyal tidak terkunci interlock. |
| `values` | `[128, 8]` | `float32` | Estimasi nilai $V(s_t)$ saat pengumpulan rollout. |
| `hidden_states` | `[128, 8, hidden_dim]` | `float32` | Keadaan rekuren tepat *sebelum* langkah dieksekusi. |
| `advantages`, `returns` | `[128, 8]` | `float32` | Diisi oleh `compute_advantages()` secara mundur via GAE($\gamma=0.99, \lambda=0.95$). |

### Pemotongan Sekuens (*Sequence Chunking*)
`iter_sequence_minibatches(chunk_len=16, num_minibatches=4)` memotong `[128, 8]` menjadi `128 / 16 × 8 = 64` potongan urutan sepanjang `16` langkah (`[64, 16, ...]`), mengambil `initial_hidden` dari indeks `t = 0, 16, 32, ...`, mengacak urutan ke-64 potongan tersebut, lalu membaginya menjadi `4` minibatch (`16` potongan per minibatch).

---

## 2. Loop Pelatihan & Masking Keputusan (`PPOTrainer`)

- **Vectorized Env**: menggunakan `gym.vector.SyncVectorEnv(..., autoreset_mode=AutoresetMode.SAME_STEP)`. Karena reset terjadi otomatis di langkah yang sama saat episode berakhir, `previous_done_mask` mengosongkan `recurrent_hidden` tepat sebelum langkah pertama episode baru:

```python
if float(previous_done_mask.sum().item()) > 0.0:
    recurrent_hidden = recurrent_hidden * (1.0 - previous_done_mask.unsqueeze(-1))
```

- **Decision Mask (`_compute_decision_mask`)**: memastikan pembaruan aktor dan entropi hanya menghitung langkah di mana kebijakan benar-benar diizinkan berpindah fase (`GREEN` dan `phase_elapsed >= min_green_steps`):

```python
not_transitioning = obs[:, 20] < 0.5
elapsed_steps = obs[:, 21] * self.env_config.max_elapsed_norm_steps
min_green_satisfied = elapsed_steps >= (self.env_config.min_green_steps - 1e-4)
return (not_transitioning & min_green_satisfied).to(dtype=torch.float32)
```

---

## 3. Fungsi Objektif PPO (`_ppo_update`)

Normalisasi *advantage*, *clipped surrogate loss*, dan *entropy* seluruhnya diberi bobot `decision_mask`, sedangkan *critic loss* dihitung penuh pada seluruh langkah agar $V(s_t)$ tetap terkalibrasi termasuk selama fase transisi `YELLOW` dan `ALL_RED`:

```python
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

total_loss = (
    policy_loss - ppo_config.ent_coef * mean_entropy + ppo_config.vf_coef * value_loss
)
```

- **Logika Resume**: saat `resume_from` diberikan, `PPOTrainer.train()` memuat `state_dict`, `optimizer_state_dict`, serta `TrainingHistory`, lalu menghitung sisa update dari `(total_timesteps - resume_timesteps) // transitions_per_update`.
- **Metrik `TrainingHistory`**: setiap akhir update mencatat `updates`, `timesteps`, `mean_episode_reward`, `mean_iqm_reward` (*interquartile mean* 25%–75% dari jendela `24` episode terakhir), `mean_throughput`, `mean_switches`, `mean_collisions`, `policy_loss`, `value_loss`, `entropy`, dan `approx_kl`.
