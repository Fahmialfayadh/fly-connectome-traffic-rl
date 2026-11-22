/**
 * Benchmark & Connectome Inspector view rendering FAFB v783 summary,
 * grouped scenario bar charts, pharmacology/lesion table, and variable reference.
 */

const AGENT_BAR_COLORS = {
  'Fixed Timer': '#8E8E93',             // Apple System Gray (Baseline)
  'PPO-MLP': '#007AFF',                 // Apple System Blue
  'PPO-GRU': '#AF52DE',                 // Apple System Purple
  'Shuffled Connectome PPO': '#FF9500', // Apple System Orange (Null Model)
  'Fly Connectome PPO': '#34C759',      // Apple System Green (Hero model)
};

const FALLBACK_BAR_PALETTE = [
  '#5AC8FA',
  '#FF3B30',
  '#30B0C7',
  '#5856D6',
  '#A2845E',
  '#FFCC00',
];

const DEFAULT_BENCHMARK_KEYS = new Set(['fixed_timer', 'mlp', 'gru', 'fly_shuffled', 'fly']);

export class BenchmarkInspectorView {
  constructor() {
    this.factsStrip = document.getElementById('connectome-facts-strip');
    this.scenarioTableBody = document.getElementById('benchmark-scenario-tbody');
    this.pharmaTableBody = document.getElementById('benchmark-pharma-tbody');
    this.rewardChartCanvas = document.getElementById('bench-reward-chart');
    this.waitChartCanvas = document.getElementById('bench-wait-chart');
    this.collisionsChartCanvas = document.getElementById('bench-collisions-chart');
    this.runBenchmarkBtn = document.getElementById('btn-run-benchmark');
    this.modelSelectorEl = document.getElementById('benchmark-model-selector');
    this.selectionCountEl = document.getElementById('benchmark-selection-count');
    this.selectAllBtn = document.getElementById('btn-bench-select-all');
    this.selectDefaultBtn = document.getElementById('btn-bench-select-default');
    this.errorNoteEl = document.getElementById('benchmark-run-error');

    this.availableModels = [];
    this.selectedKeys = new Set();
    this.hasInitializedSelection = false;
    this.isRunning = false;

    this.onRunBenchmarkRequested = null;
    this.runBenchmarkBtn.addEventListener('click', () => {
      const selected = this.getSelectedAgentKeys();
      if (selected.length === 0) {
        this.setError('Pilih minimal satu model terlebih dahulu.');
        return;
      }
      this.setError(null);
      if (this.onRunBenchmarkRequested) {
        this.onRunBenchmarkRequested(selected);
      }
    });

    if (this.selectAllBtn) {
      this.selectAllBtn.addEventListener('click', () => {
        this.selectedKeys = new Set(this.availableModels.map((m) => m.key));
        this._syncSelectorDom();
      });
    }

    if (this.selectDefaultBtn) {
      this.selectDefaultBtn.addEventListener('click', () => {
        const defaults = this.availableModels
          .filter((m) => DEFAULT_BENCHMARK_KEYS.has(m.key))
          .map((m) => m.key);
        this.selectedKeys = new Set(defaults.length > 0 ? defaults : this.availableModels.map((m) => m.key));
        this._syncSelectorDom();
      });
    }
  }

  getSelectedAgentKeys() {
    return this.availableModels
      .filter((m) => this.selectedKeys.has(m.key))
      .map((m) => m.key);
  }

  setError(message) {
    if (!this.errorNoteEl) return;
    if (!message) {
      this.errorNoteEl.hidden = true;
      this.errorNoteEl.textContent = '';
    } else {
      this.errorNoteEl.hidden = false;
      this.errorNoteEl.textContent = message;
    }
  }

  setRunningState(isRunning) {
    this.isRunning = Boolean(isRunning);
    this._updateRunButtonState();
    this.runBenchmarkBtn.textContent = isRunning ? 'Running Benchmark...' : 'Re-Run Benchmark';
  }

  _updateRunButtonState() {
    if (!this.runBenchmarkBtn) return;
    const count = this.getSelectedAgentKeys().length;
    this.runBenchmarkBtn.disabled = this.isRunning || count === 0;
  }

  renderOverview(overviewPayload, benchmarkSummary) {
    this._renderConnectomeFacts(overviewPayload ? overviewPayload.connectome_metadata : null);
    if (overviewPayload && overviewPayload.checkpoints) {
      this.renderModelSelector(overviewPayload.checkpoints);
    }
    this.renderBenchmarkSummary(benchmarkSummary);
  }

  renderModelSelector(checkpointsMap) {
    if (!this.modelSelectorEl || !checkpointsMap) return;

    const entries = Object.values(checkpointsMap).filter(
      (info) => info && (info.exists || info.key === 'fly' || info.key === 'fly_shuffled')
    );
    this.availableModels = entries;

    if (!this.hasInitializedSelection) {
      const defaults = entries
        .filter((info) => DEFAULT_BENCHMARK_KEYS.has(info.key) && info.exists)
        .map((info) => info.key);
      this.selectedKeys = new Set(defaults.length > 0 ? defaults : entries.map((e) => e.key));
      this.hasInitializedSelection = true;
    } else {
      const validKeys = new Set(entries.map((e) => e.key));
      this.selectedKeys = new Set([...this.selectedKeys].filter((k) => validKeys.has(k)));
    }

    this.modelSelectorEl.innerHTML = entries
      .map((info) => {
        const checked = this.selectedKeys.has(info.key);
        const metaText =
          info.key === 'fixed_timer'
            ? 'Rule-based (0 params)'
            : !info.exists
              ? 'Belum dilatih (untrained weights)'
              : `${Number(info.trained_timesteps || 0).toLocaleString()} steps`;
        return `
          <label class="benchmark-model-option ${checked ? 'is-selected' : ''}" data-bench-key="${info.key}">
            <input type="checkbox" value="${info.key}" ${checked ? 'checked' : ''}>
            <span class="benchmark-model-info">
              <span class="benchmark-model-name">${info.label}</span>
              <span class="benchmark-model-meta mono-value">${metaText}</span>
            </span>
          </label>
        `;
      })
      .join('');

    this.modelSelectorEl.querySelectorAll('input[type="checkbox"]').forEach((input) => {
      input.addEventListener('change', () => {
        if (input.checked) {
          this.selectedKeys.add(input.value);
        } else {
          this.selectedKeys.delete(input.value);
        }
        this.setError(null);
        this._syncSelectorDom();
      });
    });

    this._syncSelectorDom();
    if (this.selectDefaultBtn) {
      const defaultCount = entries.filter((m) => DEFAULT_BENCHMARK_KEYS.has(m.key) && m.exists).length;
      this.selectDefaultBtn.textContent = `Default (${defaultCount || entries.length})`;
    }
  }

  _syncSelectorDom() {
    if (!this.modelSelectorEl) return;
    this.modelSelectorEl.querySelectorAll('.benchmark-model-option').forEach((labelEl) => {
      const key = labelEl.dataset.benchKey;
      const checked = this.selectedKeys.has(key);
      labelEl.classList.toggle('is-selected', checked);
      const cb = labelEl.querySelector('input[type="checkbox"]');
      if (cb) cb.checked = checked;
    });
    const count = this.getSelectedAgentKeys().length;
    if (this.selectionCountEl) {
      this.selectionCountEl.textContent = `${count} dari ${this.availableModels.length} model dipilih`;
    }
    this._updateRunButtonState();
  }

  renderBenchmarkSummary(summary) {
    if (!summary) {
      this.scenarioTableBody.innerHTML =
        '<tr><td colspan="9" class="empty-cell">Belum ada data benchmark. Klik Re-Run Benchmark.</td></tr>';
      this.pharmaTableBody.innerHTML = '';
      return;
    }

    const scenarioRows = summary.scenario_results || [];
    const pharmaRows = summary.pharmacology_results || [];

    this.scenarioTableBody.innerHTML = scenarioRows
      .map(
        (row) => `
      <tr>
        <td class="mono-value">${row.scenario}</td>
        <td><strong>${row.agent_name}</strong></td>
        <td class="mono-value">${row.mean_reward.toFixed(3)} +/- ${row.std_reward.toFixed(2)}</td>
        <td class="mono-value">${row.mean_queue.toFixed(2)}</td>
        <td class="mono-value">${row.mean_wait_time_s.toFixed(2)} +/- ${row.std_wait_time_s.toFixed(2)}</td>
        <td class="mono-value">${row.mean_throughput.toFixed(1)}</td>
        <td class="mono-value">${row.mean_switches.toFixed(1)}</td>
        <td class="mono-value">${Number(row.mean_collisions ?? 0).toFixed(2)}</td>
        <td class="mono-value">${row.inference_ms_per_step.toFixed(3)}</td>
      </tr>
    `
      )
      .join('');

    this.pharmaTableBody.innerHTML = pharmaRows
      .map(
        (row) => `
      <tr>
        <td>${row.condition}</td>
        <td class="mono-value">${row.g_exc.toFixed(2)}</td>
        <td class="mono-value">${row.g_inh.toFixed(2)}</td>
        <td class="mono-value">${row.active_neurons}</td>
        <td class="mono-value">${row.mean_reward.toFixed(3)}</td>
        <td class="mono-value">${row.mean_wait_time_s.toFixed(2)}</td>
        <td class="mono-value">${row.mean_throughput.toFixed(1)}</td>
        <td class="mono-value">${Number(row.mean_collisions ?? 0).toFixed(2)}</td>
      </tr>
    `
      )
      .join('');

    this._drawGroupedBarChart(this.rewardChartCanvas, scenarioRows, 'mean_reward');
    this._drawGroupedBarChart(this.waitChartCanvas, scenarioRows, 'mean_wait_time_s');
    if (this.collisionsChartCanvas) {
      this._drawGroupedBarChart(this.collisionsChartCanvas, scenarioRows, 'mean_collisions');
    }
  }

  _renderConnectomeFacts(meta) {
    if (!meta) {
      this.factsStrip.innerHTML = '';
      return;
    }

    const items = [
      ['Total Neurons', `${meta.num_neurons} (CX=${meta.cx_count}, MB=${meta.mb_count})`, 'Total neuron terseleksi dari Drosophila FAFB v783 (Central Complex dan Mushroom Body).'],
      ['Excitatory Edges (ACh)', `${Number(meta.exc_edge_count).toLocaleString()}`, 'Jumlah koneksi sinaptik kolinergik (asetilkolin) berbobot positif.'],
      ['Inhibitory Edges (GABA/Glu)', `${Number(meta.inh_edge_count).toLocaleString()}`, 'Jumlah koneksi sinaptik GABAergik dan glutamatergik berbobot negatif.'],
      ['CX <-> MB Bridges', `${Number(meta.cross_region_edges).toLocaleString()}`, 'Koneksi sinaps inter-regional yang menghubungkan Central Complex dengan Mushroom Body.'],
      ['Matrix Sparsity', `${(meta.sparsity * 100).toFixed(4)}%`, 'Persentase pasangan neuron bernilai nol (tidak terhubung) dalam matriks ketetanggaan 2.500 x 2.500.'],
      ['Spectral Radius (rho)', `${meta.raw_spectral_radius.toFixed(2)} -> ${meta.scaled_spectral_radius.toFixed(4)}`, 'Nilai eigen dominan matriks sebelum dan sesudah penskalaan stabilitas dinamika rekuren.'],
    ];

    this.factsStrip.innerHTML = items
      .map(
        ([label, val, tooltip]) => `
      <div class="metric-cell" data-tooltip="${tooltip}" data-tooltip-pos="top">
        <span class="metric-label tooltip-trigger">${label}</span>
        <span class="metric-number mono-value">${val}</span>
      </div>
    `
      )
      .join('');
  }

  _drawGroupedBarChart(canvas, rows, metricKey) {
    const ctx = canvas.getContext('2d');
    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, w, h);

    if (rows.length === 0) {
      return;
    }

    const scenarios = [...new Set(rows.map((r) => r.scenario))];
    const agents = [...new Set(rows.map((r) => r.agent_name))];
    const lookup = new Map(rows.map((r) => [`${r.scenario}|${r.agent_name}`, r[metricKey]]));

    const values = rows.map((r) => Number(r[metricKey]));
    const minVal = Math.min(0, ...values);
    const maxVal = Math.max(0, ...values);
    const span = Math.max(maxVal - minVal, 1e-3);

    const pad = { left: 48, right: 14, top: 28, bottom: 32 };
    const plotW = w - pad.left - pad.right;
    const plotH = h - pad.top - pad.bottom;

    const zeroY = pad.top + ((maxVal - 0) / span) * plotH;

    // Horizontal grid & axis
    ctx.strokeStyle = '#e5e7eb';
    ctx.lineWidth = 1;
    ctx.fillStyle = '#52525b';
    ctx.font = '400 10px "IBM Plex Mono", monospace';
    for (let g = 0; g <= 4; g += 1) {
      const y = pad.top + (g / 4) * plotH;
      const labelVal = maxVal - (g / 4) * span;
      ctx.beginPath();
      ctx.moveTo(pad.left, y);
      ctx.lineTo(pad.left + plotW, y);
      ctx.stroke();
      ctx.fillText(labelVal.toFixed(1), 4, y + 3);
    }

    const getColor = (agentName, idx) =>
      AGENT_BAR_COLORS[agentName] || FALLBACK_BAR_PALETTE[idx % FALLBACK_BAR_PALETTE.length];

    // Legend across top (wraps cleanly if many models are selected)
    let legendX = pad.left;
    let legendY = 6;
    ctx.font = '500 10px "IBM Plex Sans", sans-serif';
    agents.forEach((agentName, idx) => {
      const shortName = agentName.replace(' Connectome PPO', ' PPO');
      const itemW = ctx.measureText(shortName).width + 24;
      if (legendX + itemW > w - pad.right && legendX > pad.left) {
        legendX = pad.left;
        legendY += 14;
      }
      ctx.fillStyle = getColor(agentName, idx);
      if (ctx.roundRect) {
        ctx.beginPath();
        ctx.roundRect(legendX, legendY, 9, 9, 2);
        ctx.fill();
      } else {
        ctx.fillRect(legendX, legendY, 9, 9);
      }
      ctx.fillStyle = '#1d1d1f';
      ctx.fillText(shortName, legendX + 13, legendY + 8);
      legendX += itemW;
    });

    const groupW = plotW / scenarios.length;
    const barW = Math.min(26, (groupW * 0.78) / Math.max(1, agents.length));

    scenarios.forEach((scenario, sIdx) => {
      const groupCenter = pad.left + (sIdx + 0.5) * groupW;
      const totalBarsW = barW * agents.length;

      agents.forEach((agentName, aIdx) => {
        const val = Number(lookup.get(`${scenario}|${agentName}`) || 0);
        const valY = pad.top + ((maxVal - val) / span) * plotH;
        const bx = groupCenter - totalBarsW / 2 + aIdx * barW;
        const by = Math.min(zeroY, valY);
        const bh = Math.max(2, Math.abs(valY - zeroY));

        ctx.fillStyle = getColor(agentName, aIdx);
        if (ctx.roundRect) {
          ctx.beginPath();
          ctx.roundRect(bx, by, Math.max(3, barW - 2), bh, [3, 3, 0, 0]);
          ctx.fill();
        } else {
          ctx.fillRect(bx, by, Math.max(3, barW - 2), bh);
        }
      });

      ctx.fillStyle = '#18181b';
      ctx.font = '600 11px "IBM Plex Mono", monospace';
      ctx.textAlign = 'center';
      ctx.fillText(scenario, groupCenter, h - 10);
      ctx.textAlign = 'left';
    });
  }
}
