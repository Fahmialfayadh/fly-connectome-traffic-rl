/**
 * Documentation tab controller handling Table of Contents scroll-spy,
 * code-block copy buttons, and dynamic syncing of evaluation tables
 * from live benchmark_summary.json with fallback to MASTER_DOC.md figures.
 */

const MASTER_DOC_SCENARIO_FALLBACK = [
  { scenario: 'normal', agent_name: 'Fly Connectome PPO', mean_reward: -35.446, mean_queue: 9.32, mean_wait_time_s: 25.82, mean_throughput: 352.4, mean_switches: 54.7, mean_collisions: 0.00, inference_ms_per_step: 0.306 },
  { scenario: 'normal', agent_name: 'Shuffled Connectome PPO', mean_reward: -35.630, mean_queue: 9.37, mean_wait_time_s: 25.99, mean_throughput: 353.2, mean_switches: 55.0, mean_collisions: 0.00, inference_ms_per_step: 0.299 },
  { scenario: 'normal', agent_name: 'PPO-GRU', mean_reward: -34.923, mean_queue: 9.32, mean_wait_time_s: 26.07, mean_throughput: 348.0, mean_switches: 53.5, mean_collisions: 0.00, inference_ms_per_step: 0.115 },
  { scenario: 'normal', agent_name: 'PPO-MLP', mean_reward: -34.605, mean_queue: 9.21, mean_wait_time_s: 25.75, mean_throughput: 349.1, mean_switches: 52.9, mean_collisions: 0.00, inference_ms_per_step: 0.106 },
  { scenario: 'normal', agent_name: 'Fixed Timer', mean_reward: -40.626, mean_queue: 11.01, mean_wait_time_s: 30.69, mean_throughput: 348.8, mean_switches: 45.0, mean_collisions: 0.00, inference_ms_per_step: 0.002 },
  { scenario: 'asymmetric', agent_name: 'Fly Connectome PPO', mean_reward: -48.370, mean_queue: 13.75, mean_wait_time_s: 23.57, mean_throughput: 571.5, mean_switches: 55.0, mean_collisions: 0.00, inference_ms_per_step: 0.336 },
  { scenario: 'asymmetric', agent_name: 'Shuffled Connectome PPO', mean_reward: -47.624, mean_queue: 13.88, mean_wait_time_s: 23.83, mean_throughput: 570.1, mean_switches: 55.0, mean_collisions: 0.00, inference_ms_per_step: 0.270 },
  { scenario: 'asymmetric', agent_name: 'PPO-GRU', mean_reward: -47.893, mean_queue: 13.34, mean_wait_time_s: 22.63, mean_throughput: 577.1, mean_switches: 53.6, mean_collisions: 0.00, inference_ms_per_step: 0.129 },
  { scenario: 'asymmetric', agent_name: 'PPO-MLP', mean_reward: -47.616, mean_queue: 13.47, mean_wait_time_s: 23.27, mean_throughput: 565.5, mean_switches: 53.5, mean_collisions: 0.00, inference_ms_per_step: 0.089 },
  { scenario: 'asymmetric', agent_name: 'Fixed Timer', mean_reward: -61.537, mean_queue: 19.12, mean_wait_time_s: 32.92, mean_throughput: 569.5, mean_switches: 45.0, mean_collisions: 0.00, inference_ms_per_step: 0.002 },
  { scenario: 'rush_hour', agent_name: 'Fly Connectome PPO', mean_reward: -79.574, mean_queue: 25.65, mean_wait_time_s: 27.96, mean_throughput: 886.7, mean_switches: 54.8, mean_collisions: 0.00, inference_ms_per_step: 0.288 },
  { scenario: 'rush_hour', agent_name: 'Shuffled Connectome PPO', mean_reward: -79.431, mean_queue: 25.60, mean_wait_time_s: 28.05, mean_throughput: 879.7, mean_switches: 54.3, mean_collisions: 0.00, inference_ms_per_step: 0.275 },
  { scenario: 'rush_hour', agent_name: 'PPO-GRU', mean_reward: -80.937, mean_queue: 26.01, mean_wait_time_s: 28.09, mean_throughput: 894.5, mean_switches: 53.9, mean_collisions: 0.00, inference_ms_per_step: 0.107 },
  { scenario: 'rush_hour', agent_name: 'PPO-MLP', mean_reward: -76.871, mean_queue: 24.71, mean_wait_time_s: 27.18, mean_throughput: 879.3, mean_switches: 54.0, mean_collisions: 0.00, inference_ms_per_step: 0.090 },
  { scenario: 'rush_hour', agent_name: 'Fixed Timer', mean_reward: -104.742, mean_queue: 35.43, mean_wait_time_s: 38.61, mean_throughput: 855.9, mean_switches: 45.0, mean_collisions: 0.00, inference_ms_per_step: 0.002 },
  { scenario: 'stress_test', agent_name: 'Fly Connectome PPO', mean_reward: -399.038, mean_queue: 144.89, mean_wait_time_s: 88.38, mean_throughput: 1448.8, mean_switches: 54.9, mean_collisions: 5.00, inference_ms_per_step: 0.293 },
  { scenario: 'stress_test', agent_name: 'Shuffled Connectome PPO', mean_reward: -395.579, mean_queue: 140.52, mean_wait_time_s: 84.97, mean_throughput: 1460.1, mean_switches: 54.9, mean_collisions: 14.00, inference_ms_per_step: 0.296 },
  { scenario: 'stress_test', agent_name: 'PPO-GRU', mean_reward: -361.209, mean_queue: 129.12, mean_wait_time_s: 78.82, mean_throughput: 1468.7, mean_switches: 52.6, mean_collisions: 8.60, inference_ms_per_step: 0.109 },
  { scenario: 'stress_test', agent_name: 'PPO-MLP', mean_reward: -381.887, mean_queue: 131.83, mean_wait_time_s: 78.69, mean_throughput: 1487.1, mean_switches: 53.8, mean_collisions: 25.20, inference_ms_per_step: 0.088 },
  { scenario: 'stress_test', agent_name: 'Fixed Timer', mean_reward: -365.067, mean_queue: 126.19, mean_wait_time_s: 76.47, mean_throughput: 1482.8, mean_switches: 45.0, mean_collisions: 20.90, inference_ms_per_step: 0.002 },
];

const MASTER_DOC_PHARMA_FALLBACK = [
  { condition: 'Baseline (g_exc=1.0, g_inh=1.0)', g_exc: 1.00, g_inh: 1.00, active_neurons: 2500, mean_reward: -79.574, mean_wait_time_s: 27.96, mean_throughput: 886.7, mean_switches: 54.8, mean_collisions: 0.00 },
  { condition: 'High Excitation (g_exc=1.8, g_inh=1.0)', g_exc: 1.80, g_inh: 1.00, active_neurons: 2500, mean_reward: -78.480, mean_wait_time_s: 27.44, mean_throughput: 885.2, mean_switches: 54.4, mean_collisions: 0.00 },
  { condition: 'Low Inhibition (g_exc=1.0, g_inh=0.2)', g_exc: 1.00, g_inh: 0.20, active_neurons: 2500, mean_reward: -78.988, mean_wait_time_s: 27.93, mean_throughput: 881.6, mean_switches: 54.7, mean_collisions: 0.00 },
  { condition: 'High Inhibition (g_exc=1.0, g_inh=2.0)', g_exc: 1.00, g_inh: 2.00, active_neurons: 2500, mean_reward: -79.107, mean_wait_time_s: 28.01, mean_throughput: 879.1, mean_switches: 54.7, mean_collisions: 0.00 },
  { condition: 'CX Lesion 25%', g_exc: 1.00, g_inh: 1.00, active_neurons: 2050, mean_reward: -80.407, mean_wait_time_s: 28.18, mean_throughput: 886.2, mean_switches: 54.6, mean_collisions: 0.00 },
  { condition: 'MB Lesion 25%', g_exc: 1.00, g_inh: 1.00, active_neurons: 2325, mean_reward: -78.734, mean_wait_time_s: 27.55, mean_throughput: 887.3, mean_switches: 54.8, mean_collisions: 0.00 },
  { condition: 'Random Lesion 25%', g_exc: 1.00, g_inh: 1.00, active_neurons: 1875, mean_reward: -79.440, mean_wait_time_s: 27.94, mean_throughput: 885.2, mean_switches: 54.6, mean_collisions: 0.00 },
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
        : 'Snapshot artifacts/benchmarks/benchmark_summary.json';
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
    const h = 305;
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

    const pad = { left: 56, right: 16, top: 44, bottom: 34 };
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

    // Legend (wrap cleanly to next line when agent labels are long)
    ctx.textAlign = 'left';
    ctx.font = '500 11px "IBM Plex Sans", sans-serif';
    let lx = pad.left;
    let ly = 8;
    const maxLegendX = w - pad.right;
    const getAgentColor = (name, idx) => {
      if (DOC_AGENT_COLORS[name]) return DOC_AGENT_COLORS[name];
      if (name.startsWith('Fly Connectome PPO')) {
        return idx === 0 ? DOC_AGENT_COLORS['Fly Connectome PPO'] : DOC_FALLBACK_COLORS[(idx - 1) % DOC_FALLBACK_COLORS.length];
      }
      if (name.startsWith('Shuffled Connectome PPO')) return DOC_AGENT_COLORS['Shuffled Connectome PPO'];
      if (name.startsWith('PPO-MLP')) return DOC_AGENT_COLORS['PPO-MLP'];
      if (name.startsWith('PPO-GRU')) return DOC_AGENT_COLORS['PPO-GRU'];
      return DOC_FALLBACK_COLORS[idx % DOC_FALLBACK_COLORS.length];
    };
    agents.forEach((name, idx) => {
      const label = name;
      const itemW = ctx.measureText(label).width + 26;
      if (lx > pad.left && lx + itemW > maxLegendX) {
        lx = pad.left;
        ly += 15;
      }
      ctx.fillStyle = getAgentColor(name, idx);
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

        ctx.fillStyle = getAgentColor(agentName, aIdx);
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
