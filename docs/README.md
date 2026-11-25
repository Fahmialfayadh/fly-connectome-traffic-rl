# Indeks Dokumentasi Teknis

Dokumentasi proyek ini disusun dalam tiga lapisan mulai dari laporan penelitian menyeluruh hingga referensi kode per modul:

## 1. Laporan Utama
- [**Technical Report (`technical_report.pdf`)**](technical_report.pdf) — Metodologi ekstraksi sub-konektom FAFB v783, *Shuffled Null Model*, lingkungan lalu lintas, arsitektur kebijakan, pelatihan PPO di CPU, hasil benchmark, dan batasan studi.

## 2. Panduan Alur & Persamaan
- [**Panduan Siklus Episode & Loop Training (`TRAINING_AND_EPISODE_GUIDE.md`)**](TRAINING_AND_EPISODE_GUIDE.md) — Diagram alur sistem, siklus 1 episode (`500` langkah / `1.000` detik), mekanisme *decision masking* PPO, dan cara kerja *Live Policy Preview*.
- [**Kamus Variabel & Pemetaan Matematika (`VARIABLES_AND_MATH.md`)**](VARIABLES_AND_MATH.md) — Pemetaan simbol persamaan biologis, observasi `obs[0:21]`, fungsi *reward*, mekanisme kolisi, dan hyperparameter PPO ke nama atribut di kode.

## 3. Referensi Implementasi & Konfigurasi
- [**Referensi Konfigurasi (`config.md`)**](config.md) — Tabel seluruh konstanta dan 5 `@dataclass(frozen=True)` pada `src/fly_traffic/config.py`.
- [**Referensi Modular Backend (`src/README.md`)**](src/README.md) — Rincian teknis per submodul: [`connectome`](src/connectome.md), [`environment`](src/environment.md), [`agents`](src/agents.md), [`training`](src/training.md), [`evaluation`](src/evaluation.md), dan [`web_ui`](src/web_ui.md).
- [**Referensi CLI & Notebook (`scripts/cli.md`)**](scripts/cli.md) — Opsi argumen skrip CLI di `scripts/`, server 3D mandiri `simulation3d/serve.py`, dan notebook `notebooks/train_agents.ipynb`.
