# Arsitektur Kebijakan & Pengendali (`src/fly_traffic/agents/`)

## 1. Perbandingan 5 Pengendali (`build_policy`, `FixedTimerAgent`)

Keempat agen berbasis neural network diseimbangkan pada anggaran parameter terlatih yang sebanding (~70 ribu parameter) agar perbandingan tetap adil pada `obs_dim = 22`, `act_dim = 4`:

| Model / Key | Kelas | Memori Rekuren (`hidden_dim`) | Parameter Terlatih | Total Parameter | Arsitektur Inti |
|---|---|---|---|---:|---|
| `fly` | `FlyConnectomePolicy` | `2.500` neuron | `70.006` | `70.006` (+`87.171` buffer *frozen*) | `ConnectomeRecurrentCell` (FAFB v783 asli) |
| `fly_shuffled` | `FlyConnectomePolicy` | `2.500` neuron | `70.006` | `70.006` (+`87.171` buffer *frozen*) | `ConnectomeRecurrentCell` (graf *global rewiring*) |
| `gru` | `GRUBaselinePolicy` | `105` unit | `69.725` | `69.725` | `Linear(22, 105) + Tanh -> GRUCell(105, 105)` |
| `mlp` | `MLPBaselinePolicy` | `1` (placeholder statis) | `69.755` | `69.755` | `Linear(22, 250) + Tanh -> Linear(250, 250) + Tanh` |
| `fixed_timer` | `FixedTimerAgent` | Tidak ada (stateless) | `0` | `0` | Siklus tetap `phase_durations = (10, 6, 10, 6)` langkah — fase lurus (`NS_STRAIGHT`, `EW_STRAIGHT`) 10 langkah (20 s), fase belok (`NS_TURN`, `EW_TURN`) 6 langkah (12 s) — diulang berurutan tanpa membaca antrean. |

*(Catatan: `MLPBaselinePolicy` menyediakan `hidden_dim = 1` dummy tensor agar berbagi API `BaseRecurrentActorCritic` yang sama dengan `PPOTrainer`, sekaligus menyimpan `_last_layer1` & `_last_layer2` saat `torch.no_grad()` untuk telemetri Web UI.)*

---

## 2. Dinamika Sel Biologis (`ConnectomeRecurrentCell`)

Matriks sinaptik `w_exc_csr` dan `w_inh_csr` didaftarkan sebagai `register_buffer(..., persistent=False)` (`requires_grad=False`), sehingga bobot konektom biologis tetap dibekukan sementara gradien BPTT tetap mengalir melewati $W_{\text{exc}}^T$ dan $W_{\text{inh}}^T$ ke adapter terlatih.

```python
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

### Parameter & Kontrol Runtime pada `FlyConnectomePolicy`

| Atribut / Parameter | Status | Rincian Perhitungan / Fungsi |
|---|---|---|
| `cell.w_exc_csr`, `cell.w_inh_csr` | *Frozen buffer* | Tensor sparse CSR `[2500, 2500]`; tidak masuk ke `state_dict` checkpoint (`persistent=False`). |
| `cell.input_projection` | **Trainable** (`57.500`) | `nn.Linear(22, 2500)` (`22 × 2500 + 2500`), inisialisasi orthogonal `gain=1.0`, bias `0`. |
| `cell.tau_logit` | **Trainable** (`1`) | Dipetakan menjadi `leak_rate = 0.05 + 0.90 * torch.sigmoid(tau_logit)` pada rentang `(0.05, 0.95)`, default awal `0.35`. |
| `actor_head` | **Trainable** (`10.004`) | `nn.Linear(2500, 4)` (`2500 × 4 + 4`), inisialisasi orthogonal `gain=0.01`. |
| `critic_head` | **Trainable** (`2.501`) | `nn.Linear(2500, 1)` (`2500 × 1 + 1`), inisialisasi orthogonal `gain=1.0`. |
| `cell.g_exc`, `cell.g_inh` | Kontrol runtime | Skalar float `>= 0.0` (default `1.0`), diubah via `set_pharmacology(g_exc, g_inh)`. |
| `cell.lesion_mask` | Kontrol runtime | Buffer biner `[1, 2500]` (diubah via `apply_random_lesion`, `apply_region_lesion`, `reset_lesion`). |

---

## 3. Inferensi & Kontrak Sekuens (`BaseRecurrentActorCritic`)

- **Langkah tunggal (`forward_step` / `get_action_and_value`)**: memproses tensor `[batch, 22]` + `[batch, hidden_dim]` dan mengembalikan `PolicyStepOutput(action, log_prob, entropy, value, next_hidden)`.
- **Sekuens BPTT (`forward_sequence`)**: meloop langkah waktu `t = 0 .. seq_len - 1`, dan mereset `hidden = torch.where(done_mask > 0.5, zeros, hidden)` setiap kali `dones_seq[t] == 1` agar status rekuren tidak bocor antar episode.
- **`NeuralAgentAdapter`**: membungkus kebijakan PyTorch untuk antarmuka `select_action(obs, deterministic)` berbasis NumPy dan mengekspos `latest_hidden_numpy`, `latest_action_probabilities`, serta `latest_value_estimate`.

---

## 4. Struktur Payload Checkpoint (`save_policy_checkpoint` / `load_policy_checkpoint`)

Setiap berkas `.pt` disimpan aman (`weights_only=True`) dengan struktur kunci:

| Kunci Payload | Tipe | Isi |
|---|---|---|
| `model_type` | `str` | `"fly"`, `"fly_shuffled"`, `"mlp"`, atau `"gru"`. |
| `obs_dim`, `act_dim` | `int` | Dimensi input dan output (`22`, `4`). |
| `trainable_params` | `int` | Jumlah parameter terlatih aktual. |
| `trained_timesteps` | `int` | Akumulasi total langkah latih (dipakai oleh fitur `--resume`). |
| `leak_rate_tau` | `float \| None` | Nilai $\tau$ terlatih pada model Fly, `None` untuk MLP/GRU. |
| `seed` | `int \| None` | Seed pelatihan/rewiring yang digunakan saat menyimpan checkpoint. |
| `saved_at_iso` | `str` | Timestamp UTC format ISO-8601. |
| `state_dict` | `dict` | Bobot adapter/backbone (`w_exc_csr`/`w_inh_csr` dimuat ulang dari artefak konektom saat `load_policy_checkpoint`). |
| `history` | `dict[str, list[float]]` | Riwayat metrik `TrainingHistory`. |
| `connectome_path` | `str \| None` | Path referensi artefak konektom yang dipakai. |
| `shuffled_connectome_path` | `str \| None` | Path artefak graf teracak spesifik seed untuk model `fly_shuffled`. |
| `optimizer_state_dict` | `dict \| None` | State optimizer Adam untuk melanjutkan pelatihan (`--resume`). |
