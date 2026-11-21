/**
 * Live telemetry renderer for non-connectome controllers (PPO-GRU, PPO-MLP, Fixed Timer).
 * Renders real-time hidden state activations, feedforward layer activations, or fixed-cycle
 * phase progress in the right column of Live Simulation so every controller has factual stats.
 */

const PHASE_NAMES = ['NS Straight', 'NS Turn', 'EW Straight', 'EW Turn'];
const MAX_SPARKLINE_POINTS = 60;

export function resolveControllerKind(frame) {
  if (!frame || typeof frame !== 'object') {
    return 'fly';
  }
  const modelTel = frame.model_telemetry || {};
  if (modelTel.kind) {
    return String(modelTel.kind).toLowerCase();
  }
  const prov = frame.checkpoint_provenance || {};
  if (prov.model_type) {
    return String(prov.model_type).toLowerCase();
  }
  const key = String(frame.agent_key || '').toLowerCase();
  const name = String(frame.agent_name || '').toLowerCase();
  if (key === 'fixed_timer' || name.includes('fixed timer')) return 'fixed_timer';
  if (key === 'fly_shuffled' || name.includes('shuffled')) return 'fly_shuffled';
  if (key === 'fly' || frame.connectome_telemetry?.is_fly_active) return 'fly';
  if (key === 'gru' || name.includes('gru')) return 'gru';
  if (key === 'mlp' || name.includes('mlp')) return 'mlp';
  return 'fixed_timer';
}

export class ModelTelemetryRenderer {
  constructor(rootElement) {
    this.root = rootElement;
    this.canvasWrap = rootElement ? rootElement.querySelector('#baseline-canvas-wrap') : null;
    this.canvas = rootElement ? rootElement.querySelector('#baseline-viz-canvas') : null;
    this.ctx = this.canvas ? this.canvas.getContext('2d') : null;
    this.scaleRow = rootElement ? rootElement.querySelector('#baseline-scale-row') : null;
    this.timerView = rootElement ? rootElement.querySelector('#baseline-timer-view') : null;
    this.readoutGrid = rootElement ? rootElement.querySelector('#baseline-readout-grid') : null;

    this.lastAgentKey = null;
    this.gruDeltaHistory = [];
    this.valueHistory = [];
  }

  render(frame) {
    if (!this.root || !frame) {
      return;
    }

    const telemetry = frame.model_telemetry || {};
    const policy = frame.policy_telemetry || {};
    const prov = frame.checkpoint_provenance || {};
    const snapshot = frame.snapshot || {};
    const kind = resolveControllerKind(frame);

    if (frame.agent_key !== this.lastAgentKey) {
      this.lastAgentKey = frame.agent_key;
      this.gruDeltaHistory = [];
      this.valueHistory = [];
    }

    if (policy.value_estimate !== null && policy.value_estimate !== undefined) {
      this.valueHistory.push(Number(policy.value_estimate));
      if (this.valueHistory.length > MAX_SPARKLINE_POINTS) {
        this.valueHistory.shift();
      }
    }

    if (kind === 'gru') {
      this._renderGru(telemetry, policy, prov);
    } else if (kind === 'mlp') {
      this._renderMlp(telemetry, policy, prov);
    } else {
      this._renderFixedTimer(telemetry, snapshot);
    }
  }

  _setVisible(el, visible) {
    if (!el) return;
    el.hidden = !visible;
    el.classList.toggle('is-hidden', !visible);
  }

  _computePolicyEntropy(probs) {
    if (!Array.isArray(probs) || probs.length === 0) return 0;
    let ent = 0;
    for (const pRaw of probs) {
      const p = Number(pRaw || 0);
      if (p > 1e-9) ent -= p * Math.log(p);
    }
    return ent;
  }

  _renderGru(t, policy, prov) {
    this._setVisible(this.canvasWrap, true);
    this._setVisible(this.timerView, false);

    const hasLiveTensor = Boolean(t.activations_b64);
    this._setVisible(this.scaleRow, hasLiveTensor);

    const delta = Number(t.delta_l2 || 0);
    const meanAbs = Number(t.mean_abs || 0);
    if (hasLiveTensor) {
      this.gruDeltaHistory.push(delta);
      if (this.gruDeltaHistory.length > MAX_SPARKLINE_POINTS) {
        this.gruDeltaHistory.shift();
      }
    }

    if (this.ctx && this.canvas) {
      if (hasLiveTensor) {
        this._drawGruCanvas(t);
      } else {
        this._drawValueFallbackCanvas(
          'PPO-GRU (105 GRU units) — Critic V(s_t) Trace',
          '#af52de',
          'Restart backend server (launch_web_lab.py) untuk streaming 105-unit h_t live'
        );
      }
    }

    const valueText =
      policy.value_estimate !== null && policy.value_estimate !== undefined
        ? Number(policy.value_estimate).toFixed(3)
        : '-';
    const probs = policy.action_probabilities || [];
    const maxConf = probs.length ? `${Math.round(Math.max(...probs) * 100)}%` : '-';
    const entropy = this._computePolicyEntropy(probs).toFixed(3);
    const params = Number(t.trainable_params || prov.trainable_params || 69725).toLocaleString();
    const steps = Number(t.trained_timesteps || prov.trained_timesteps || 0).toLocaleString();

    if (hasLiveTensor) {
      const satPct = `${(Number(t.saturated_ratio || 0) * 100).toFixed(1)}%`;
      const posPct = `${(Number(t.positive_ratio || 0) * 100).toFixed(1)}%`;
      this._renderReadouts([
        ['Architecture', `${t.hidden_dim || 105}-unit GRUCell`, 'Sel rekuren GRU dengan input encoder 22 ke 105 unit.'],
        ['Trainable Params', params, 'Total parameter terlatih PPO-GRU (setara anggaran ~70k parameter).'],
        ['Mean |h_t|', meanAbs.toFixed(4), 'Rata-rata magnitudo absolut seluruh 105 unit state tersembunyi GRU saat ini.'],
        ['Std(h_t)', Number(t.std || 0).toFixed(4), 'Simpangan baku aktivasi h_t antar unit GRU pada langkah aktif.'],
        ['||h_t||_2', Number(t.l2_norm || 0).toFixed(3), 'Norma Euclidean vektor state tersembunyi h_t.'],
        ['||dh_t||_2', delta.toFixed(4), 'Besar perubahan state rekuren dibanding langkah sebelumnya (||h_t - h_{t-1}||_2).'],
        ['Active > 0 / |h|>0.8', `${posPct} / ${satPct}`, 'Persentase unit bernilai positif dan persentase unit mendekati saturasi (|h_t| > 0.8).'],
        ['Critic V(s_t)', valueText, 'Estimasi nilai keadaan V(s_t) oleh critic head dari h_t.'],
      ]);
    } else {
      this._renderReadouts([
        ['Architecture', '105-unit GRUCell', 'Sel rekuren GRU dengan input encoder 22 ke 105 unit.'],
        ['Recurrent Memory', 'Yes (105-dim h_t)', 'Mempertahankan vektor state rekuren lintas langkah simulasi.'],
        ['Trainable Params', params, 'Total parameter terlatih PPO-GRU.'],
        ['Trained Steps', steps, 'Jumlah langkah pelatihan PPO yang tercatat di checkpoint.'],
        ['Checkpoint', prov.checkpoint_file || '-', 'Berkas bobot model aktif di artifacts/checkpoints/.'],
        ['Policy Entropy', entropy, 'Entropi distribusi aksi softmax saat ini (nats).'],
        ['Top Action Conf.', maxConf, 'Probabilitas tertinggi dari distribusi 4 fasa sinyal.'],
        ['Critic V(s_t)', valueText, 'Estimasi nilai keadaan V(s_t) oleh critic head.'],
      ]);
    }
  }

  _renderMlp(t, policy, prov) {
    this._setVisible(this.canvasWrap, true);
    this._setVisible(this.timerView, false);

    const hasLiveTensor = Boolean(t.layer1_b64 && t.layer2_b64);
    this._setVisible(this.scaleRow, hasLiveTensor);

    if (this.ctx && this.canvas) {
      if (hasLiveTensor) {
        this._drawMlpCanvas(t);
      } else {
        this._drawValueFallbackCanvas(
          'PPO-MLP (2x250 Tanh) — Critic V(s_t) Trace',
          '#007aff',
          'Restart backend server (launch_web_lab.py) untuk streaming aktivasi 250+250 unit live'
        );
      }
    }

    const width = Number(t.hidden_width || 250);
    const valueText =
      policy.value_estimate !== null && policy.value_estimate !== undefined
        ? Number(policy.value_estimate).toFixed(3)
        : '-';
    const probs = policy.action_probabilities || [];
    const maxConf = probs.length ? `${Math.round(Math.max(...probs) * 100)}%` : '-';
    const entropy = this._computePolicyEntropy(probs).toFixed(3);
    const params = Number(t.trainable_params || prov.trainable_params || 69755).toLocaleString();
    const steps = Number(t.trained_timesteps || prov.trained_timesteps || 0).toLocaleString();

    if (hasLiveTensor) {
      const l1Sat = `${(Number(t.layer1_saturated_ratio || 0) * 100).toFixed(1)}%`;
      const l2Sat = `${(Number(t.layer2_saturated_ratio || 0) * 100).toFixed(1)}%`;
      this._renderReadouts([
        ['Architecture', `22 -> ${width} -> ${width} -> 4`, 'Jaringan feedforward 2 lapis tersembunyi dengan aktivasi Tanh.'],
        ['Recurrent Memory', 'None (Feedforward)', 'MLP mengambil keputusan hanya dari observasi x_t saat ini tanpa memori antar-step.'],
        ['Layer 1 Mean |a|', Number(t.layer1_mean_abs || 0).toFixed(4), 'Rata-rata magnitudo aktivasi Tanh pada hidden layer pertama (250 unit).'],
        ['Layer 1 |a| > 0.8', l1Sat, 'Proporsi unit pada layer pertama yang mendekati batas saturasi Tanh.'],
        ['Layer 2 Mean |a|', Number(t.layer2_mean_abs || 0).toFixed(4), 'Rata-rata magnitudo aktivasi Tanh pada hidden layer kedua sebelum actor/critic head.'],
        ['Layer 2 |a| > 0.8', l2Sat, 'Proporsi unit pada layer kedua yang mendekati batas saturasi Tanh.'],
        ['Trainable Params', params, 'Total parameter terlatih PPO-MLP (setara anggaran ~70k parameter).'],
        ['Critic V(s_t)', valueText, 'Estimasi nilai keadaan V(s_t) oleh critic head.'],
      ]);
    } else {
      this._renderReadouts([
        ['Architecture', `22 -> ${width} -> ${width} -> 4`, 'Jaringan feedforward 2 lapis tersembunyi dengan aktivasi Tanh.'],
        ['Recurrent Memory', 'None (Feedforward)', 'MLP mengambil keputusan hanya dari observasi x_t saat ini tanpa memori antar-step.'],
        ['Trainable Params', params, 'Total parameter terlatih PPO-MLP.'],
        ['Trained Steps', steps, 'Jumlah langkah pelatihan PPO yang tercatat di checkpoint.'],
        ['Checkpoint', prov.checkpoint_file || '-', 'Berkas bobot model aktif di artifacts/checkpoints/.'],
        ['Policy Entropy', entropy, 'Entropi distribusi aksi softmax saat ini (nats).'],
        ['Top Action Conf.', maxConf, 'Probabilitas tertinggi dari distribusi 4 fasa sinyal.'],
        ['Critic V(s_t)', valueText, 'Estimasi nilai keadaan V(s_t) oleh critic head.'],
      ]);
    }
  }

  _renderFixedTimer(t, snapshot) {
    this._setVisible(this.canvasWrap, false);
    this._setVisible(this.scaleRow, false);
    this._setVisible(this.timerView, true);

    const durations = Array.isArray(t.phase_durations) ? t.phase_durations : [8, 4, 8, 4];
    const activePhase = Number(t.active_phase ?? snapshot.active_phase ?? 0);
    const nextPhase = Number(t.next_phase ?? (activePhase + 1) % 4);
    const elapsed = Number(t.phase_elapsed ?? snapshot.phase_elapsed ?? 0);
    const stage = String(t.signal_stage ?? snapshot.signal_stage ?? 'GREEN');
    const dt = Number(t.dt_seconds || 2.0);
    const target = Number(t.phase_target_steps ?? durations[activePhase % 4] ?? 8);
    const remaining = Number(t.remaining_steps ?? Math.max(0, target - elapsed));
    const cycleTotal = Number(t.cycle_total_steps ?? durations.reduce((a, b) => a + b, 0));

    if (this.timerView) {
      this.timerView.innerHTML = `
        <div class="timer-cycle-header">
          <span class="timer-cycle-title">Cyclic Phase Schedule (${durations.join('-')} steps)</span>
          <span class="timer-stage-pill stage-${stage.toLowerCase()} mono-value">${stage}</span>
        </div>
        <div class="timer-phase-list">
          ${durations
            .map((dur, idx) => {
              const isActive = idx === activePhase;
              const isNext = idx === nextPhase;
              const ratio = isActive ? Math.min(1, elapsed / Math.max(1, dur)) : 0;
              const statusTag = isActive
                ? `${elapsed} / ${dur} steps`
                : isNext
                  ? 'Next'
                  : `${dur} steps (${(dur * dt).toFixed(0)}s)`;
              return `
                <div class="timer-phase-row ${isActive ? 'is-active' : ''} ${isNext ? 'is-next' : ''}">
                  <div class="timer-phase-top">
                    <span class="timer-phase-name">${idx}: ${PHASE_NAMES[idx]}</span>
                    <span class="timer-phase-meta mono-value">${statusTag}</span>
                  </div>
                  <div class="timer-phase-track">
                    <div class="timer-phase-fill" style="transform: scaleX(${ratio.toFixed(4)})"></div>
                  </div>
                </div>
              `;
            })
            .join('')}
        </div>
      `;
    }

    this._renderReadouts([
      ['Active Phase', PHASE_NAMES[activePhase] || String(activePhase), 'Fasa sinyal yang sedang dijalankan oleh aturan siklus tetap.'],
      ['Next Scheduled', PHASE_NAMES[nextPhase] || String(nextPhase), 'Fasa berikutnya dalam urutan tetap (0 -> 1 -> 2 -> 3).'],
      ['Elapsed / Target', `${elapsed} / ${target} steps`, 'Jumlah langkah hijau yang telah berjalan dibanding durasi tetap fasa aktif.'],
      ['Remaining Green', `${remaining} steps (${(remaining * dt).toFixed(1)} s)`, 'Sisa langkah sebelum pengontrol berpindah ke fasa berikutnya.'],
      ['Signal Stage', stage, 'Tahap interlock sinyal saat ini (GREEN, YELLOW, atau ALL_RED).'],
      ['Full Cycle', `${cycleTotal} steps (${(cycleTotal * dt).toFixed(0)} s)`, 'Total durasi satu putaran penuh empat fasa (8 + 4 + 8 + 4 langkah).'],
      ['Trainable Params', '0 (Rule-Based)', 'Pengendali deterministik berbasis aturan tanpa bobot yang dilatih.'],
      ['Memory', 'Stateless (Phase Timer)', 'Keputusan hanya bergantung pada penghitung durasi fasa dari observasi.'],
    ]);
  }

  _renderReadouts(rows) {
    if (!this.readoutGrid) {
      return;
    }
    this.readoutGrid.innerHTML = rows
      .map(
        ([key, val, tooltip]) => `
        <div class="readout-row"${tooltip ? ` data-tooltip="${tooltip}" data-tooltip-pos="left"` : ''}>
          <span class="readout-key${tooltip ? ' tooltip-trigger' : ''}">${key}</span>
          <span class="readout-val mono-value">${val}</span>
        </div>
      `
      )
      .join('');
  }

  _drawValueFallbackCanvas(title, accentColor, footerHint) {
    const ctx = this.ctx;
    const w = this.canvas.width;
    const h = this.canvas.height;
    ctx.clearRect(0, 0, w, h);

    ctx.fillStyle = '#0f131a';
    ctx.fillRect(0, 0, w, h);

    const padX = 16;
    const plotTop = 38;
    const plotH = h - 78;
    const plotW = w - padX * 2;

    ctx.fillStyle = '#e2e8f0';
    ctx.font = '600 11px "IBM Plex Mono", monospace';
    ctx.fillText(title, padX, 22);

    ctx.fillStyle = '#161b24';
    ctx.fillRect(padX, plotTop, plotW, plotH);
    ctx.strokeStyle = 'rgba(148, 163, 184, 0.22)';
    ctx.lineWidth = 1;
    ctx.strokeRect(padX + 0.5, plotTop + 0.5, plotW - 1, plotH - 1);

    const series = this.valueHistory;
    if (series.length >= 2) {
      let minV = Math.min(...series);
      let maxV = Math.max(...series);
      if (Math.abs(maxV - minV) < 0.1) {
        minV -= 0.5;
        maxV += 0.5;
      }
      ctx.strokeStyle = accentColor;
      ctx.lineWidth = 2;
      ctx.beginPath();
      series.forEach((v, idx) => {
        const px = padX + (idx / Math.max(1, MAX_SPARKLINE_POINTS - 1)) * plotW;
        const py = plotTop + plotH - 10 - ((v - minV) / (maxV - minV)) * (plotH - 20);
        if (idx === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
      });
      ctx.stroke();

      ctx.fillStyle = '#94a3b8';
      ctx.font = '500 10px "IBM Plex Mono", monospace';
      ctx.fillText(`min ${minV.toFixed(2)}`, padX + 8, plotTop + plotH - 8);
      ctx.textAlign = 'right';
      ctx.fillText(`max ${maxV.toFixed(2)}`, padX + plotW - 8, plotTop + 16);
      ctx.textAlign = 'left';
    } else {
      ctx.fillStyle = '#94a3b8';
      ctx.font = '500 11px "IBM Plex Sans", sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('Collecting live Critic V(s_t) samples...', w / 2, plotTop + plotH / 2);
      ctx.textAlign = 'left';
    }

    ctx.fillStyle = '#94a3b8';
    ctx.font = '500 10px "IBM Plex Sans", sans-serif';
    ctx.fillText(footerHint, padX, h - 14);
  }

  _drawGruCanvas(t) {
    const ctx = this.ctx;
    const w = this.canvas.width;
    const h = this.canvas.height;
    ctx.clearRect(0, 0, w, h);

    ctx.fillStyle = '#0f131a';
    ctx.fillRect(0, 0, w, h);

    const bytes = t.activations_b64 ? this._decodeBase64(t.activations_b64) : new Uint8Array(105).fill(128);
    const cols = 15;
    const rows = Math.ceil(bytes.length / cols);

    const padX = 16;
    const headerY = 20;
    const gridTop = 30;
    const gridH = 128;
    const gridW = w - padX * 2;
    const cellW = gridW / cols;
    const cellH = gridH / rows;

    ctx.fillStyle = '#cbd5e1';
    ctx.font = '600 11px "IBM Plex Mono", monospace';
    ctx.fillText(`GRU Hidden State h_t (${bytes.length} units, 15x${rows})`, padX, headerY);

    for (let i = 0; i < bytes.length; i += 1) {
      const r = Math.floor(i / cols);
      const c = i % cols;
      const val = (bytes[i] / 255) * 2 - 1;
      const [cr, cg, cb] = this._divergingRgb(val);
      ctx.fillStyle = `rgb(${cr},${cg},${cb})`;
      ctx.fillRect(
        padX + c * cellW + 1,
        gridTop + r * cellH + 1,
        Math.max(1, cellW - 2),
        Math.max(1, cellH - 2)
      );
    }

    const sparkTop = gridTop + gridH + 26;
    const sparkH = h - sparkTop - 16;
    ctx.fillStyle = '#cbd5e1';
    ctx.font = '600 10.5px "IBM Plex Mono", monospace';
    ctx.fillText('Recurrent State Delta ||h_t - h_{t-1}||_2 (last 60 steps)', padX, sparkTop - 8);

    ctx.fillStyle = '#161b24';
    ctx.fillRect(padX, sparkTop, gridW, sparkH);
    ctx.strokeStyle = 'rgba(148, 163, 184, 0.22)';
    ctx.lineWidth = 1;
    ctx.strokeRect(padX + 0.5, sparkTop + 0.5, gridW - 1, sparkH - 1);

    const series = this.gruDeltaHistory;
    if (series.length >= 2) {
      const maxVal = Math.max(0.25, ...series);
      ctx.strokeStyle = '#af52de';
      ctx.lineWidth = 2;
      ctx.beginPath();
      series.forEach((val, idx) => {
        const px = padX + (idx / Math.max(1, MAX_SPARKLINE_POINTS - 1)) * gridW;
        const py = sparkTop + sparkH - 6 - (val / maxVal) * (sparkH - 12);
        if (idx === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
      });
      ctx.stroke();

      ctx.fillStyle = '#94a3b8';
      ctx.font = '500 10px "IBM Plex Mono", monospace';
      ctx.textAlign = 'right';
      ctx.fillText(`max ${maxVal.toFixed(2)}`, padX + gridW - 6, sparkTop + 13);
      ctx.textAlign = 'left';
    }
  }

  _drawMlpCanvas(t) {
    const ctx = this.ctx;
    const w = this.canvas.width;
    const h = this.canvas.height;
    ctx.clearRect(0, 0, w, h);

    ctx.fillStyle = '#0f131a';
    ctx.fillRect(0, 0, w, h);

    const width = Number(t.hidden_width || 250);
    const l1 = t.layer1_b64 ? this._decodeBase64(t.layer1_b64) : new Uint8Array(width).fill(128);
    const l2 = t.layer2_b64 ? this._decodeBase64(t.layer2_b64) : new Uint8Array(width).fill(128);

    const padX = 16;
    const cols = 25;
    const rows = Math.ceil(width / cols);
    const blockW = w - padX * 2;
    const blockH = 92;

    const drawLayerBlock = (label, bytes, topY) => {
      ctx.fillStyle = '#cbd5e1';
      ctx.font = '600 11px "IBM Plex Mono", monospace';
      ctx.fillText(label, padX, topY - 7);

      const cellW = blockW / cols;
      const cellH = blockH / rows;
      for (let i = 0; i < bytes.length; i += 1) {
        const r = Math.floor(i / cols);
        const c = i % cols;
        const val = (bytes[i] / 255) * 2 - 1;
        const [cr, cg, cb] = this._divergingRgb(val);
        ctx.fillStyle = `rgb(${cr},${cg},${cb})`;
        ctx.fillRect(
          padX + c * cellW + 0.8,
          topY + r * cellH + 0.8,
          Math.max(1, cellW - 1.6),
          Math.max(1, cellH - 1.6)
        );
      }
    };

    drawLayerBlock(`Hidden Layer 1 — Tanh (${l1.length} units, 25x${rows})`, l1, 28);
    drawLayerBlock(`Hidden Layer 2 — Tanh (${l2.length} units, 25x${rows})`, l2, 146);
  }

  _divergingRgb(value) {
    const base = [30, 36, 48];
    const positive = [235, 118, 46];
    const negative = [37, 99, 235];
    const t = Math.min(Math.abs(value), 1);
    const target = value >= 0 ? positive : negative;
    return [
      Math.round(base[0] + t * (target[0] - base[0])),
      Math.round(base[1] + t * (target[1] - base[1])),
      Math.round(base[2] + t * (target[2] - base[2])),
    ];
  }

  _decodeBase64(b64) {
    const binary = atob(b64);
    const output = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i += 1) {
      output[i] = binary.charCodeAt(i);
    }
    return output;
  }
}
