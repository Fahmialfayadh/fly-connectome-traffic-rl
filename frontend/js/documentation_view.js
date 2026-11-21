/**
 * Documentation tab controller handling Table of Contents scroll-spy,
 * code-block copy buttons, and dynamic syncing of evaluation tables
 * from live benchmark_summary.json with fallback to MASTER_DOC.md figures.
 */

const MASTER_DOC_SCENARIO_FALLBACK = [
  { scenario: 'normal', agent_name: 'Fixed Timer', mean_reward: -17.908, mean_queue: 8.22, mean_wait_time_s: 22.47, mean_throughput: 209.9, mean_switches: 37.0, mean_collisions: 0.10, inference_ms_per_step: 0.002 },
  { scenario: 'normal', agent_name: 'PPO-MLP', mean_reward: -10.582, mean_queue: 4.97, mean_wait_time_s: 13.79, mean_throughput: 211.0, mean_switches: 70.7, mean_collisions: 0.00, inference_ms_per_step: 0.097 },
  { scenario: 'normal', agent_name: 'PPO-GRU', mean_reward: -12.341, mean_queue: 5.78, mean_wait_time_s: 15.83, mean_throughput: 213.3, mean_switches: 74.2, mean_collisions: 0.00, inference_ms_per_step: 0.184 },
  { scenario: 'normal', agent_name: 'Fly Connectome PPO', mean_reward: -11.427, mean_queue: 5.27, mean_wait_time_s: 14.48, mean_throughput: 213.6, mean_switches: 67.6, mean_collisions: 0.00, inference_ms_per_step: 0.542 },
  { scenario: 'asymmetric', agent_name: 'Fixed Timer', mean_reward: -30.564, mean_queue: 15.56, mean_wait_time_s: 27.09, mean_throughput: 329.3, mean_switches: 37.0, mean_collisions: 1.70, inference_ms_per_step: 0.005 },
  { scenario: 'asymmetric', agent_name: 'PPO-MLP', mean_reward: -16.787, mean_queue: 8.35, mean_wait_time_s: 14.65, mean_throughput: 332.4, mean_switches: 71.9, mean_collisions: 0.10, inference_ms_per_step: 0.167 },
  { scenario: 'asymmetric', agent_name: 'PPO-GRU', mean_reward: -21.074, mean_queue: 9.67, mean_wait_time_s: 16.78, mean_throughput: 331.4, mean_switches: 68.0, mean_collisions: 2.00, inference_ms_per_step: 0.221 },
  { scenario: 'asymmetric', agent_name: 'Fly Connectome PPO', mean_reward: -16.585, mean_queue: 8.44, mean_wait_time_s: 14.70, mean_throughput: 335.4, mean_switches: 61.9, mean_collisions: 0.10, inference_ms_per_step: 0.599 },
  { scenario: 'rush_hour', agent_name: 'Fixed Timer', mean_reward: -30.049, mean_queue: 15.39, mean_wait_time_s: 26.26, mean_throughput: 323.5, mean_switches: 37.0, mean_collisions: 1.30, inference_ms_per_step: 0.005 },
  { scenario: 'rush_hour', agent_name: 'PPO-MLP', mean_reward: -18.594, mean_queue: 9.40, mean_wait_time_s: 16.11, mean_throughput: 338.7, mean_switches: 71.4, mean_collisions: 0.20, inference_ms_per_step: 0.121 },
  { scenario: 'rush_hour', agent_name: 'PPO-GRU', mean_reward: -20.442, mean_queue: 9.87, mean_wait_time_s: 17.31, mean_throughput: 327.3, mean_switches: 71.2, mean_collisions: 0.70, inference_ms_per_step: 0.135 },
  { scenario: 'rush_hour', agent_name: 'Fly Connectome PPO', mean_reward: -18.483, mean_queue: 9.19, mean_wait_time_s: 16.26, mean_throughput: 328.5, mean_switches: 66.8, mean_collisions: 0.10, inference_ms_per_step: 0.608 },
];

const MASTER_DOC_PHARMA_FALLBACK = [
  { condition: 'Baseline (g_exc=1.0, g_inh=1.0)', g_exc: 1.00, g_inh: 1.00, active_neurons: 2500, mean_reward: -18.483, mean_wait_time_s: 16.26, mean_throughput: 328.5, mean_switches: 66.8, mean_collisions: 0.10 },
  { condition: 'High Excitation (g_exc=1.8)', g_exc: 1.80, g_inh: 1.00, active_neurons: 2500, mean_reward: -18.282, mean_wait_time_s: 15.88, mean_throughput: 333.8, mean_switches: 64.4, mean_collisions: 0.30 },
  { condition: 'Low Inhibition (g_inh=0.2)', g_exc: 1.00, g_inh: 0.20, active_neurons: 2500, mean_reward: -18.333, mean_wait_time_s: 16.03, mean_throughput: 334.8, mean_switches: 67.2, mean_collisions: 0.30 },
  { condition: 'High Inhibition (g_inh=2.0)', g_exc: 1.00, g_inh: 2.00, active_neurons: 2500, mean_reward: -18.822, mean_wait_time_s: 16.54, mean_throughput: 331.8, mean_switches: 67.9, mean_collisions: 0.30 },
  { condition: 'CX Lesion 25%', g_exc: 1.00, g_inh: 1.00, active_neurons: 2050, mean_reward: -18.915, mean_wait_time_s: 16.38, mean_throughput: 339.1, mean_switches: 66.8, mean_collisions: 0.30 },
  { condition: 'MB Lesion 25%', g_exc: 1.00, g_inh: 1.00, active_neurons: 2325, mean_reward: -18.753, mean_wait_time_s: 16.47, mean_throughput: 331.7, mean_switches: 67.4, mean_collisions: 0.20 },
  { condition: 'Random Lesion 25%', g_exc: 1.00, g_inh: 1.00, active_neurons: 1875, mean_reward: -18.021, mean_wait_time_s: 15.85, mean_throughput: 330.4, mean_switches: 66.5, mean_collisions: 0.20 },
];

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

    this.observer = null;
    this.isVisible = false;
    this._onScroll = this._handleScroll.bind(this);

    if (this.panel) {
      this._bindTocClicks();
      this._bindCodeCopyButtons();
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
}
