/**
 * Documentation tab controller handling Table of Contents scroll-spy,
 * code-block copy buttons, and dynamic syncing of evaluation tables
 * from live benchmark_summary.json with fallback to MASTER_DOC.md figures.
 */

const MASTER_DOC_SCENARIO_FALLBACK = [
  { scenario: 'normal', agent_name: 'Fixed Timer', mean_reward: -17.688, mean_queue: 8.06, mean_wait_time_s: 22.33, mean_throughput: 208.2, mean_switches: 37.0, mean_collisions: 0.00, inference_ms_per_step: 0.002 },
  { scenario: 'normal', agent_name: 'PPO-MLP', mean_reward: -10.731, mean_queue: 5.03, mean_wait_time_s: 13.90, mean_throughput: 211.9, mean_switches: 71.0, mean_collisions: 0.00, inference_ms_per_step: 0.075 },
  { scenario: 'normal', agent_name: 'PPO-GRU', mean_reward: -12.010, mean_queue: 5.62, mean_wait_time_s: 15.50, mean_throughput: 212.7, mean_switches: 74.0, mean_collisions: 0.00, inference_ms_per_step: 0.165 },
  { scenario: 'normal', agent_name: 'Fly Connectome PPO', mean_reward: -11.477, mean_queue: 5.29, mean_wait_time_s: 14.50, mean_throughput: 213.6, mean_switches: 68.0, mean_collisions: 0.00, inference_ms_per_step: 0.189 },
  { scenario: 'asymmetric', agent_name: 'Fixed Timer', mean_reward: -31.220, mean_queue: 15.93, mean_wait_time_s: 28.14, mean_throughput: 328.0, mean_switches: 37.0, mean_collisions: 0.00, inference_ms_per_step: 0.001 },
  { scenario: 'asymmetric', agent_name: 'PPO-MLP', mean_reward: -16.544, mean_queue: 8.27, mean_wait_time_s: 14.30, mean_throughput: 337.3, mean_switches: 72.2, mean_collisions: 0.00, inference_ms_per_step: 0.068 },
  { scenario: 'asymmetric', agent_name: 'PPO-GRU', mean_reward: -19.962, mean_queue: 9.21, mean_wait_time_s: 15.88, mean_throughput: 338.5, mean_switches: 69.0, mean_collisions: 0.00, inference_ms_per_step: 0.077 },
  { scenario: 'asymmetric', agent_name: 'Fly Connectome PPO', mean_reward: -16.792, mean_queue: 8.55, mean_wait_time_s: 15.09, mean_throughput: 332.5, mean_switches: 62.8, mean_collisions: 0.00, inference_ms_per_step: 0.198 },
  { scenario: 'rush_hour', agent_name: 'Fixed Timer', mean_reward: -29.838, mean_queue: 15.26, mean_wait_time_s: 26.29, mean_throughput: 324.7, mean_switches: 37.0, mean_collisions: 0.00, inference_ms_per_step: 0.001 },
  { scenario: 'rush_hour', agent_name: 'PPO-MLP', mean_reward: -18.950, mean_queue: 9.55, mean_wait_time_s: 16.65, mean_throughput: 329.8, mean_switches: 71.5, mean_collisions: 0.00, inference_ms_per_step: 0.065 },
  { scenario: 'rush_hour', agent_name: 'PPO-GRU', mean_reward: -22.309, mean_queue: 10.85, mean_wait_time_s: 18.35, mean_throughput: 339.5, mean_switches: 70.3, mean_collisions: 0.00, inference_ms_per_step: 0.106 },
  { scenario: 'rush_hour', agent_name: 'Fly Connectome PPO', mean_reward: -18.269, mean_queue: 9.16, mean_wait_time_s: 16.02, mean_throughput: 332.3, mean_switches: 67.1, mean_collisions: 0.00, inference_ms_per_step: 0.473 },
  { scenario: 'stress_test', agent_name: 'Fixed Timer', mean_reward: -241.743, mean_queue: 140.88, mean_wait_time_s: 85.60, mean_throughput: 770.6, mean_switches: 37.0, mean_collisions: 42.50, inference_ms_per_step: 0.005 },
  { scenario: 'stress_test', agent_name: 'PPO-MLP', mean_reward: -317.427, mean_queue: 185.54, mean_wait_time_s: 136.41, mean_throughput: 546.7, mean_switches: 73.2, mean_collisions: 41.90, inference_ms_per_step: 0.182 },
  { scenario: 'stress_test', agent_name: 'PPO-GRU', mean_reward: -272.365, mean_queue: 150.45, mean_wait_time_s: 116.43, mean_throughput: 544.7, mean_switches: 59.6, mean_collisions: 55.60, inference_ms_per_step: 0.299 },
  { scenario: 'stress_test', agent_name: 'Fly Connectome PPO', mean_reward: -320.557, mean_queue: 191.08, mean_wait_time_s: 143.97, mean_throughput: 521.5, mean_switches: 71.3, mean_collisions: 42.80, inference_ms_per_step: 0.486 },
];

const MASTER_DOC_PHARMA_FALLBACK = [
  { condition: 'Baseline (g_exc=1.0, g_inh=1.0)', g_exc: 1.00, g_inh: 1.00, active_neurons: 2500, mean_reward: -18.269, mean_wait_time_s: 16.02, mean_throughput: 332.3, mean_switches: 67.1, mean_collisions: 0.00 },
  { condition: 'High Excitation (g_exc=1.8)', g_exc: 1.80, g_inh: 1.00, active_neurons: 2500, mean_reward: -18.635, mean_wait_time_s: 15.99, mean_throughput: 338.6, mean_switches: 65.0, mean_collisions: 0.00 },
  { condition: 'Low Inhibition (g_inh=0.2)', g_exc: 1.00, g_inh: 0.20, active_neurons: 2500, mean_reward: -18.763, mean_wait_time_s: 16.43, mean_throughput: 331.1, mean_switches: 67.4, mean_collisions: 0.00 },
  { condition: 'High Inhibition (g_inh=2.0)', g_exc: 1.00, g_inh: 2.00, active_neurons: 2500, mean_reward: -18.365, mean_wait_time_s: 16.29, mean_throughput: 331.8, mean_switches: 67.3, mean_collisions: 0.00 },
  { condition: 'CX Lesion 25%', g_exc: 1.00, g_inh: 1.00, active_neurons: 2050, mean_reward: -19.169, mean_wait_time_s: 16.50, mean_throughput: 340.7, mean_switches: 67.6, mean_collisions: 0.00 },
  { condition: 'MB Lesion 25%', g_exc: 1.00, g_inh: 1.00, active_neurons: 2325, mean_reward: -18.656, mean_wait_time_s: 16.49, mean_throughput: 330.2, mean_switches: 67.6, mean_collisions: 0.00 },
  { condition: 'Random Lesion 25%', g_exc: 1.00, g_inh: 1.00, active_neurons: 1875, mean_reward: -18.264, mean_wait_time_s: 16.17, mean_throughput: 330.2, mean_switches: 66.8, mean_collisions: 0.00 },
];

const DOC_AGENT_COLORS = {
  'Fixed Timer': '#8E8E93',
  'PPO-MLP': '#007AFF',
  'PPO-GRU': '#AF52DE',
  'Shuffled Connectome PPO': '#FF9500',
  'Fly Connectome PPO': '#34C759',
};

const DOC_FALLBACK_COLORS = ['#5AC8FA', '#FF3B30', '#30B0C7', '#5856D6', '#A2845E'];

export class DocumentationView {
  constructor(panelElement) {
    this.panel = panelElement;
    this.tocLinks = panelElement ? Array.from(panelElement.querySelectorAll('.doc-toc-link')) : [];
    this.sections = panelElement ? Array.from(panelElement.querySelectorAll('.doc-section[id]')) : [];
    this.scenarioTbody = panelElement ? panelElement.querySelector('#doc-bench-scenario-tbody') : null;
    this.pharmaTbody = panelElement ? panelElement.querySelector('#doc-bench-pharma-tbody') : null;
    this.sourceBadge = panelElement ? panelElement.querySelector('#doc-bench-source-badge') : null;
    this.progressFill = panelElement ? panelElement.querySelector('#doc-read-progress') : null;
    this.backToTopBtn = panelElement ? panelElement.querySelector('#btn-doc-top') : null;
    this.scenarioChartCanvas = panelElement ? panelElement.querySelector('#doc-scenario-chart') : null;
    this.pharmaChartCanvas = panelElement ? panelElement.querySelector('#doc-pharma-chart') : null;
    this.scenarioMetricTabs = panelElement ? panelElement.querySelector('#doc-scenario-metric-tabs') : null;
    this.pharmaMetricTabs = panelElement ? panelElement.querySelector('#doc-pharma-metric-tabs') : null;

    this.activeScenarioMetric = 'mean_reward';
    this.activePharmaMetric = 'mean_reward';
    this.currentScenarioRows = MASTER_DOC_SCENARIO_FALLBACK;
    this.currentPharmaRows = MASTER_DOC_PHARMA_FALLBACK;

    this.observer = null;
    this.isVisible = false;
    this._onScroll = this._handleScroll.bind(this);

    if (this.panel) {
      this._bindTocClicks();
      this._bindCodeCopyButtons();
      this._bindChartTabs();
      if (this.backToTopBtn) {
        this.backToTopBtn.addEventListener('click', () => {
          const prefersReduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
          window.scrollTo({ top: 0, behavior: prefersReduced ? 'auto' : 'smooth' });
        });
      }
      this.syncBenchmarkSummary(null);
    }
  }

  setVisible(visible) {
    const next = Boolean(visible);
    if (this.isVisible === next) {
      return;
    }
    this.isVisible = next;
    if (next) {
      this._startScrollSpy();
      window.addEventListener('scroll', this._onScroll, { passive: true });
      this._handleScroll();
      this._drawScenarioChart();
      this._drawPharmaChart();
    } else {
      this._stopScrollSpy();
      window.removeEventListener('scroll', this._onScroll);
    }
  }

  _handleScroll() {
    if (!this.isVisible || !this.panel) return;
    const rect = this.panel.getBoundingClientRect();
    const scrollable = Math.max(1, this.panel.scrollHeight - window.innerHeight + 120);
    const scrolled = Math.min(Math.max(-rect.top, 0), scrollable);
    const ratio = scrolled / scrollable;
    if (this.progressFill) {
      this.progressFill.style.transform = `scaleX(${ratio.toFixed(4)})`;
    }
    // Also update active TOC reliably even on fast scrolls
    let activeId = this.sections[0]?.id || null;
    for (const sec of this.sections) {
      if (sec.getBoundingClientRect().top <= 130) {
        activeId = sec.id;
      }
    }
    if (activeId) {
      this._setActiveToc(activeId);
    }
  }

  syncBenchmarkSummary(summary) {
    const hasLiveScenario =
      summary && Array.isArray(summary.scenario_results) && summary.scenario_results.length > 0;
    const hasLivePharma =
      summary && Array.isArray(summary.pharmacology_results) && summary.pharmacology_results.length > 0;

    const scenarioRows = hasLiveScenario ? summary.scenario_results : MASTER_DOC_SCENARIO_FALLBACK;
    const pharmaRows = hasLivePharma ? summary.pharmacology_results : MASTER_DOC_PHARMA_FALLBACK;
    this.currentScenarioRows = scenarioRows;
    this.currentPharmaRows = pharmaRows;

    if (this.sourceBadge) {
      this.sourceBadge.textContent = hasLiveScenario
        ? 'Disinkronkan dari artifacts/benchmarks/benchmark_summary.json'
        : 'Snapshot laporan docs/MASTER_DOC.md';
    }

    if (this.scenarioTbody) {
      this.scenarioTbody.innerHTML = scenarioRows
        .map(
          (r) => `
          <tr>
            <td class="mono-value">${r.scenario}</td>
            <td><strong>${r.agent_name}</strong></td>
            <td class="mono-value num-cell">${Number(r.mean_reward).toFixed(3)}</td>
            <td class="mono-value num-cell">${Number(r.mean_queue).toFixed(2)}</td>
            <td class="mono-value num-cell">${Number(r.mean_wait_time_s).toFixed(2)}</td>
            <td class="mono-value num-cell">${Number(r.mean_throughput).toFixed(1)}</td>
            <td class="mono-value num-cell">${Number(r.mean_switches).toFixed(1)}</td>
            <td class="mono-value num-cell">${Number(r.mean_collisions ?? 0).toFixed(2)}</td>
            <td class="mono-value num-cell">${Number(r.inference_ms_per_step).toFixed(3)}</td>
          </tr>
        `
        )
        .join('');
    }

    if (this.pharmaTbody) {
      this.pharmaTbody.innerHTML = pharmaRows
        .map(
          (r) => `
          <tr>
            <td>${r.condition}</td>
            <td class="mono-value num-cell">${Number(r.g_exc).toFixed(2)}</td>
            <td class="mono-value num-cell">${Number(r.g_inh).toFixed(2)}</td>
            <td class="mono-value num-cell">${r.active_neurons}</td>
            <td class="mono-value num-cell">${Number(r.mean_reward).toFixed(3)}</td>
            <td class="mono-value num-cell">${Number(r.mean_wait_time_s).toFixed(2)}</td>
            <td class="mono-value num-cell">${Number(r.mean_throughput).toFixed(1)}</td>
            <td class="mono-value num-cell">${Number(r.mean_switches ?? 0).toFixed(1)}</td>
            <td class="mono-value num-cell">${Number(r.mean_collisions ?? 0).toFixed(2)}</td>
          </tr>
        `
        )
        .join('');
    }

    this._drawScenarioChart();
    this._drawPharmaChart();
  }

  _bindTocClicks() {
    this.tocLinks.forEach((link) => {
      link.addEventListener('click', (event) => {
        const targetId = link.getAttribute('href')?.replace(/^#/, '');
        if (!targetId) return;
        const target = document.getElementById(targetId);
        if (!target) return;
        event.preventDefault();
        const prefersReduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
        target.scrollIntoView({
          behavior: prefersReduced ? 'auto' : 'smooth',
          block: 'start',
        });
        if (window.history && typeof window.history.replaceState === 'function') {
          window.history.replaceState({ tab: 'docs' }, '', `/docs#${targetId}`);
        }
        this._setActiveToc(targetId);
      });
    });
  }

  _bindCodeCopyButtons() {
    const buttons = this.panel.querySelectorAll('.doc-code-copy');
    buttons.forEach((btn) => {
      btn.addEventListener('click', async () => {
        const pre = btn.closest('.doc-code-block')?.querySelector('pre code');
        if (!pre) return;
        const originalText = btn.textContent;
        try {
          await navigator.clipboard.writeText(pre.textContent || '');
          btn.textContent = 'Copied';
          setTimeout(() => {
            btn.textContent = originalText;
          }, 1400);
        } catch {
          btn.textContent = 'Copy';
        }
      });
    });
  }

  _startScrollSpy() {
    if (this.observer || !this.sections.length || typeof IntersectionObserver === 'undefined') {
      return;
    }
    this.observer = new IntersectionObserver(
      (entries) => {
        const visible = entries
          .filter((e) => e.isIntersecting)
          .sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top);
        if (visible.length > 0) {
          this._setActiveToc(visible[0].target.id);
        }
      },
      {
        rootMargin: '-88px 0px -65% 0px',
        threshold: 0,
      }
    );
    this.sections.forEach((sec) => this.observer.observe(sec));
  }

  _stopScrollSpy() {
    if (this.observer) {
      this.observer.disconnect();
      this.observer = null;
    }
  }

  _setActiveToc(activeId) {
    this.tocLinks.forEach((link) => {
      const hrefId = link.getAttribute('href')?.replace(/^#/, '');
      link.classList.toggle('is-active', hrefId === activeId);
    });
  }

  _bindChartTabs() {
    if (this.scenarioMetricTabs) {
      this.scenarioMetricTabs.querySelectorAll('.doc-chart-tab').forEach((btn) => {
        btn.addEventListener('click', () => {
          this.activeScenarioMetric = btn.dataset.metric || 'mean_reward';
          this.scenarioMetricTabs.querySelectorAll('.doc-chart-tab').forEach((b) => {
            b.classList.toggle('is-active', b === btn);
          });
          this._drawScenarioChart();
        });
      });
    }

    if (this.pharmaMetricTabs) {
      this.pharmaMetricTabs.querySelectorAll('.doc-chart-tab').forEach((btn) => {
        btn.addEventListener('click', () => {
          this.activePharmaMetric = btn.dataset.metric || 'mean_reward';
          this.pharmaMetricTabs.querySelectorAll('.doc-chart-tab').forEach((b) => {
            b.classList.toggle('is-active', b === btn);
          });
          this._drawPharmaChart();
        });
      });
    }
  }

  _prepareHiDpiCanvas(canvas, logicalW, logicalH) {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = logicalW * dpr;
    canvas.height = logicalH * dpr;
    const ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, logicalW, logicalH);
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, logicalW, logicalH);
    return ctx;
  }

  _drawScenarioChart() {
    if (!this.scenarioChartCanvas) return;
    const rows = this.currentScenarioRows || [];
    const w = 760;
    const h = 290;
    const ctx = this._prepareHiDpiCanvas(this.scenarioChartCanvas, w, h);
    if (!rows.length) return;

    const metricKey = this.activeScenarioMetric;
    const scenarios = [...new Set(rows.map((r) => r.scenario))];
    const agents = [...new Set(rows.map((r) => r.agent_name))];
    const lookup = new Map(rows.map((r) => [`${r.scenario}|${r.agent_name}`, Number(r[metricKey] ?? 0)]));

    const values = rows.map((r) => Number(r[metricKey] ?? 0));
    const minVal = Math.min(0, ...values);
    const maxVal = Math.max(0, ...values);
    const span = Math.max(maxVal - minVal, 1e-3);

    const pad = { left: 56, right: 16, top: 34, bottom: 34 };
    const plotW = w - pad.left - pad.right;
    const plotH = h - pad.top - pad.bottom;
    const zeroY = pad.top + ((maxVal - 0) / span) * plotH;

    // Gridlines
    ctx.strokeStyle = '#eceef2';
    ctx.lineWidth = 1;
    ctx.fillStyle = '#6e6e73';
    ctx.font = '400 10.5px "IBM Plex Mono", monospace';
    ctx.textAlign = 'right';
    for (let g = 0; g <= 4; g += 1) {
      const y = pad.top + (g / 4) * plotH;
      const labelVal = maxVal - (g / 4) * span;
      ctx.beginPath();
      ctx.moveTo(pad.left, y);
      ctx.lineTo(pad.left + plotW, y);
      ctx.stroke();
      ctx.fillText(Math.abs(labelVal) >= 100 ? labelVal.toFixed(0) : labelVal.toFixed(1), pad.left - 8, y + 3.5);
    }

    // Zero baseline
    ctx.strokeStyle = '#d1d5db';
    ctx.beginPath();
    ctx.moveTo(pad.left, zeroY);
    ctx.lineTo(pad.left + plotW, zeroY);
    ctx.stroke();

    const getColor = (name, idx) => DOC_AGENT_COLORS[name] || DOC_FALLBACK_COLORS[idx % DOC_FALLBACK_COLORS.length];

    // Legend
    ctx.textAlign = 'left';
    ctx.font = '500 11px "IBM Plex Sans", sans-serif';
    let lx = pad.left;
    const ly = 10;
    agents.forEach((name, idx) => {
      const label = name;
      const itemW = ctx.measureText(label).width + 26;
      ctx.fillStyle = getColor(name, idx);
      if (ctx.roundRect) {
        ctx.beginPath();
        ctx.roundRect(lx, ly, 10, 10, 2);
        ctx.fill();
      } else {
        ctx.fillRect(lx, ly, 10, 10);
      }
      ctx.fillStyle = '#1d1d1f';
      ctx.fillText(label, lx + 14, ly + 9);
      lx += itemW;
    });

    // Grouped bars
    const groupW = plotW / Math.max(1, scenarios.length);
    const barW = Math.min(30, (groupW * 0.72) / Math.max(1, agents.length));

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
        const radius = val >= 0 ? [3, 3, 0, 0] : [0, 0, 3, 3];
        if (ctx.roundRect) {
          ctx.beginPath();
          ctx.roundRect(bx, by, Math.max(4, barW - 3), bh, radius);
          ctx.fill();
        } else {
          ctx.fillRect(bx, by, Math.max(4, barW - 3), bh);
        }
      });

      ctx.fillStyle = '#18181b';
      ctx.font = '600 11.5px "IBM Plex Mono", monospace';
      ctx.textAlign = 'center';
      ctx.fillText(scenario, groupCenter, h - 10);
    });
    ctx.textAlign = 'left';
  }

  _drawPharmaChart() {
    if (!this.pharmaChartCanvas) return;
    const rows = this.currentPharmaRows || [];
    const w = 760;
    const h = 250;
    const ctx = this._prepareHiDpiCanvas(this.pharmaChartCanvas, w, h);
    if (!rows.length) return;

    const metricKey = this.activePharmaMetric;
    const values = rows.map((r) => Number(r[metricKey] ?? 0));
    const isNegativeMetric = values.every((v) => v <= 0);
    const minVal = isNegativeMetric ? Math.min(...values) * 1.12 : 0;
    const maxVal = isNegativeMetric ? 0 : Math.max(...values) * 1.12;
    const span = Math.max(maxVal - minVal, 1e-3);

    const pad = { left: 215, right: 64, top: 18, bottom: 26 };
    const plotW = w - pad.left - pad.right;
    const plotH = h - pad.top - pad.bottom;
    const rowH = plotH / rows.length;
    const barH = Math.min(18, rowH * 0.62);

    // Vertical gridlines
    ctx.strokeStyle = '#eceef2';
    ctx.lineWidth = 1;
    ctx.fillStyle = '#6e6e73';
    ctx.font = '400 10px "IBM Plex Mono", monospace';
    ctx.textAlign = 'center';
    for (let g = 0; g <= 4; g += 1) {
      const x = pad.left + (g / 4) * plotW;
      const tick = minVal + (g / 4) * span;
      ctx.beginPath();
      ctx.moveTo(x, pad.top);
      ctx.lineTo(x, pad.top + plotH);
      ctx.stroke();
      ctx.fillText(tick.toFixed(1), x, h - 8);
    }

    // Baseline reference line
    const baseVal = Number(rows[0][metricKey] ?? 0);
    const baseX = pad.left + ((baseVal - minVal) / span) * plotW;
    ctx.save();
    ctx.setLineDash([4, 3]);
    ctx.strokeStyle = '#9ca3af';
    ctx.beginPath();
    ctx.moveTo(baseX, pad.top);
    ctx.lineTo(baseX, pad.top + plotH);
    ctx.stroke();
    ctx.restore();

    rows.forEach((row, idx) => {
      const val = Number(row[metricKey] ?? 0);
      const cy = pad.top + (idx + 0.5) * rowH;
      const by = cy - barH / 2;

      const zeroX = pad.left + ((0 - minVal) / span) * plotW;
      const valX = pad.left + ((val - minVal) / span) * plotW;
      const bx = Math.min(zeroX, valX);
      const bw = Math.max(3, Math.abs(valX - zeroX));

      const isBaseline = idx === 0;
      const isLesion = row.condition.toLowerCase().includes('lesion');
      ctx.fillStyle = isBaseline ? '#34C759' : isLesion ? '#FF9500' : '#007AFF';

      if (ctx.roundRect) {
        ctx.beginPath();
        ctx.roundRect(bx, by, bw, barH, 3);
        ctx.fill();
      } else {
        ctx.fillRect(bx, by, bw, barH);
      }

      // Condition label
      ctx.fillStyle = '#18181b';
      ctx.font = isBaseline ? '600 11px "IBM Plex Sans", sans-serif' : '400 11px "IBM Plex Sans", sans-serif';
      ctx.textAlign = 'right';
      ctx.fillText(row.condition, pad.left - 12, cy + 4);

      // Value label
      ctx.fillStyle = '#3f3f46';
      ctx.font = '500 10.5px "IBM Plex Mono", monospace';
      ctx.textAlign = 'left';
      const formatted = metricKey === 'mean_reward' ? val.toFixed(3) : metricKey === 'mean_wait_time_s' ? `${val.toFixed(2)} s` : val.toFixed(1);
      ctx.fillText(formatted, Math.max(valX, zeroX) + 8, cy + 3.5);
    });
    ctx.textAlign = 'left';
  }
}
