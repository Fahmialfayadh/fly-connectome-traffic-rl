/**
 * Benchmark & Connectome Inspector view rendering FAFB v783 summary,
 * grouped scenario bar charts, pharmacology/lesion table, and variable reference.
 */

const AGENT_BAR_COLORS = {
  'Fixed Timer': '#8E8E93',       // Apple System Gray (Baseline)
  'PPO-MLP': '#007AFF',           // Apple System Blue
  'PPO-GRU': '#AF52DE',           // Apple System Purple
  'Fly Connectome PPO': '#34C759', // Apple System Green (Hero model)
};

export class BenchmarkInspectorView {
  constructor() {
    this.factsStrip = document.getElementById('connectome-facts-strip');
    this.scenarioTableBody = document.getElementById('benchmark-scenario-tbody');
    this.pharmaTableBody = document.getElementById('benchmark-pharma-tbody');
    this.rewardChartCanvas = document.getElementById('bench-reward-chart');
    this.waitChartCanvas = document.getElementById('bench-wait-chart');
    this.runBenchmarkBtn = document.getElementById('btn-run-benchmark');

    this.onRunBenchmarkRequested = null;
    this.runBenchmarkBtn.addEventListener('click', () => {
      if (this.onRunBenchmarkRequested) {
        this.onRunBenchmarkRequested();
      }
    });
  }

  setRunningState(isRunning) {
    this.runBenchmarkBtn.disabled = isRunning;
    this.runBenchmarkBtn.textContent = isRunning ? 'Running Benchmark...' : 'Re-Run Benchmark';
  }

  renderOverview(overviewPayload, benchmarkSummary) {
    this._renderConnectomeFacts(overviewPayload ? overviewPayload.connectome_metadata : null);
    this.renderBenchmarkSummary(benchmarkSummary);
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
      ['Matrix Sparsity', `${(meta.sparsity * 100).toFixed(4)}%`, 'Persentase elemen non-zero dalam matriks adjacency konektom.'],
      ['Spectral Radius (rho)', `${meta.raw_spectral_radius.toFixed(2)} -> ${meta.scaled_spectral_radius.toFixed(4)}`, 'Nilai eigen dominan matriks sebelum dan sesudah penskalaan stabilitas dinamika reservoir.'],
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

    // Legend across top
    agents.forEach((agentName, idx) => {
      const lx = pad.left + idx * 116;
      ctx.fillStyle = AGENT_BAR_COLORS[agentName] || '#475569';
      if (ctx.roundRect) {
        ctx.beginPath();
        ctx.roundRect(lx, 6, 9, 9, 2);
        ctx.fill();
      } else {
        ctx.fillRect(lx, 6, 9, 9);
      }
      ctx.fillStyle = '#1d1d1f';
      ctx.font = '500 10px "IBM Plex Sans", sans-serif';
      ctx.fillText(agentName.replace(' Connectome PPO', ' PPO'), lx + 13, 14);
    });

    const groupW = plotW / scenarios.length;
    const barW = Math.min(26, (groupW * 0.75) / agents.length);

    scenarios.forEach((scenario, sIdx) => {
      const groupCenter = pad.left + (sIdx + 0.5) * groupW;
      const totalBarsW = barW * agents.length;

      agents.forEach((agentName, aIdx) => {
        const val = Number(lookup.get(`${scenario}|${agentName}`) || 0);
        const valY = pad.top + ((maxVal - val) / span) * plotH;
        const bx = groupCenter - totalBarsW / 2 + aIdx * barW;
        const by = Math.min(zeroY, valY);
        const bh = Math.max(2, Math.abs(valY - zeroY));

        ctx.fillStyle = AGENT_BAR_COLORS[agentName] || '#475569';
        if (ctx.roundRect) {
          ctx.beginPath();
          ctx.roundRect(bx, by, barW - 2, bh, [3, 3, 0, 0]);
          ctx.fill();
        } else {
          ctx.fillRect(bx, by, barW - 2, bh);
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
