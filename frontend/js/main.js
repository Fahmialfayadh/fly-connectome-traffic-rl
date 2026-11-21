/**
 * Top-level frontend application controller managing tab navigation,
 * simulation/training WebSockets, and Light Instrument Lab UI state.
 */

import { BenchmarkInspectorView } from './benchmark_view.js?v=20261121_13';
import { Brain3DRenderer, webglAvailable } from './brain3d_renderer.js?v=20261121_13';
import { ConnectomeRenderer } from './connectome_renderer.js?v=20261121_13';
import { DocumentationView } from './documentation_view.js?v=20261121_13';
import { IntersectionRenderer } from './intersection_renderer.js?v=20261121_13';
import { ModelTelemetryRenderer, resolveControllerKind } from './model_telemetry_renderer.js?v=20261121_13';
import { TrainingStudioView } from './training_studio.js?v=20261121_13';
import { ArchitectureView } from './architecture_view.js?v=20261121_13';

const PHASE_NAMES = ['NS Straight', 'NS Turn', 'EW Straight', 'EW Turn'];

const TAB_ROUTES = {
  simulation: '/simulation',
  training: '/training',
  benchmark: '/benchmark',
  architecture: '/architecture',
  docs: '/docs',
};

const ROUTE_TO_TAB = {
  '/': 'simulation',
  '/simulation': 'simulation',
  '/training': 'training',
  '/benchmark': 'benchmark',
  '/architecture': 'architecture',
  '/docs': 'docs',
  '/documentation': 'docs',
};

const TAB_TITLES = {
  simulation: 'Live Simulation — Drosophila Connectome Traffic Lab',
  training: 'Training Studio — Drosophila Connectome Traffic Lab',
  benchmark: 'Benchmark & Connectome — Drosophila Connectome Traffic Lab',
  architecture: '3D Architecture — Drosophila Connectome Traffic Lab',
  docs: 'Documentation — Drosophila Connectome Traffic Lab',
};

class TrafficLabApp {
  constructor() {
    this.intersectionRenderer = new IntersectionRenderer(
      document.getElementById('intersection-canvas')
    );
    this.connectomeRenderer = new ConnectomeRenderer(
      document.getElementById('connectome-canvas')
    );
    this.brain3dRoot = document.getElementById('brain3d-root');
    this.brain3dRenderer =
      this.brain3dRoot && webglAvailable()
        ? new Brain3DRenderer(this.brain3dRoot)
        : null;
    this.connectomeViewMode = this.brain3dRenderer ? '3d' : '2d';
    this.isFlyModelActive = true;
    this.modelTelemetryRenderer = new ModelTelemetryRenderer(
      document.getElementById('baseline-panel-body')
    );
    this.trainingStudio = new TrainingStudioView();
    this.benchmarkView = new BenchmarkInspectorView();
    const archPanel = document.getElementById('panel-architecture');
    this.architectureView = archPanel && webglAvailable()
      ? new ArchitectureView(archPanel)
      : null;
    this.documentationView = new DocumentationView(
      document.getElementById('panel-docs')
    );

    this.activeTab = 'simulation';
    this.isTrainingActive = false;
    this.latestSimFrame = null;
    this.checkpointsInventory = null;
    this.lastInventoryAgentKey = null;

    this.simSocket = null;
    this.trainSocket = null;
    this.animationFrameId = null;

    this._initTabs();
    this._bindBrain3dControls();
    this._bindSimulationControls();
    this._bindStudioAndBenchmarkCallbacks();
  }

  async init() {
    const initialTab = this._tabFromPathname(window.location.pathname);
    this._activateTab(initialTab, { updateHistory: false });
    if (window.location.hash) {
      const hashTarget = document.getElementById(window.location.hash.slice(1));
      if (hashTarget) {
        requestAnimationFrame(() => hashTarget.scrollIntoView({ behavior: 'auto', block: 'start' }));
      }
    }

    this._setConnectomeViewMode(this.connectomeViewMode);
    if (this.brain3dRenderer) {
      this.brain3dRenderer
        .init()
        .then(() => {
          if (this.latestSimFrame) {
            this.brain3dRenderer.applyFrame(this.latestSimFrame);
          }
        })
        .catch((err) => {
          console.error('3D Brain initialization failed, falling back to 2D:', err);
          this.brain3dRenderer = null;
          this._setConnectomeViewMode('2d');
        });
    }

    await this._fetchOverview();
    this._connectSimulationSocket();
    this._connectTrainingSocket();

    const animate = (timestampMs) => {
      this.intersectionRenderer.renderFrame(timestampMs, this.activeTab === 'simulation');
      this.animationFrameId = requestAnimationFrame(animate);
    };
    this.animationFrameId = requestAnimationFrame(animate);
  }

  _setConnectomeViewMode(mode) {
    const resolved = mode === '3d' && this.brain3dRenderer ? '3d' : '2d';
    this.connectomeViewMode = resolved;

    const btn3d = document.getElementById('btn-view-3d');
    const btn2d = document.getElementById('btn-view-2d');
    const frame2d = document.getElementById('connectome-2d-frame');
    const summaryRow = document.getElementById('connectome-summary-row');

    if (btn3d) {
      btn3d.classList.toggle('is-active', resolved === '3d');
      btn3d.setAttribute('aria-pressed', resolved === '3d' ? 'true' : 'false');
      btn3d.disabled = !this.brain3dRenderer;
    }
    if (btn2d) {
      btn2d.classList.toggle('is-active', resolved === '2d');
      btn2d.setAttribute('aria-pressed', resolved === '2d' ? 'true' : 'false');
    }
    if (this.brain3dRoot) {
      this.brain3dRoot.classList.toggle('is-hidden', resolved !== '3d');
    }
    if (frame2d) {
      frame2d.classList.toggle('is-hidden', resolved !== '2d');
    }
    if (summaryRow) {
      summaryRow.classList.toggle('is-3d-mode', resolved === '3d');
    }
    if (this.brain3dRenderer) {
      this.brain3dRenderer.setVisible(
        resolved === '3d' && this.activeTab === 'simulation' && this.isFlyModelActive
      );
      if (resolved === '3d' && this.isFlyModelActive && this.latestSimFrame) {
        this.brain3dRenderer.applyFrame(this.latestSimFrame);
      }
    }
    if (resolved === '2d' && this.isFlyModelActive && this.latestSimFrame) {
      this.connectomeRenderer.render(
        this.latestSimFrame.connectome_telemetry || {},
        this.latestSimFrame.agent_name
      );
    }
  }

  _bindBrain3dControls() {
    const btn3d = document.getElementById('btn-view-3d');
    const btn2d = document.getElementById('btn-view-2d');
    if (btn3d) btn3d.addEventListener('click', () => this._setConnectomeViewMode('3d'));
    if (btn2d) btn2d.addEventListener('click', () => this._setConnectomeViewMode('2d'));

    if (!this.brain3dRoot) return;

    const filterButtons = Array.from(
      this.brain3dRoot.querySelectorAll('[data-brain3d-filter]')
    );
    filterButtons.forEach((btn) => {
      btn.addEventListener('click', () => {
        if (!this.brain3dRenderer) return;
        this.brain3dRenderer.setFilterMode(btn.dataset.brain3dFilter);
        filterButtons.forEach((b) => {
          const active = b === btn;
          b.classList.toggle('is-active', active);
          b.setAttribute('aria-pressed', active ? 'true' : 'false');
        });
      });
    });

    const layoutButtons = Array.from(
      this.brain3dRoot.querySelectorAll('[data-brain3d-layout]')
    );
    layoutButtons.forEach((btn) => {
      btn.addEventListener('click', () => {
        if (!this.brain3dRenderer) return;
        const mode = btn.dataset.brain3dLayout;
        this.brain3dRenderer.setLayout(mode);
        layoutButtons.forEach((b) => {
          const active = b === btn;
          b.classList.toggle('is-active', active);
          b.setAttribute('aria-pressed', active ? 'true' : 'false');
        });
      });
    });

    const focusButtons = Array.from(
      this.brain3dRoot.querySelectorAll('[data-brain3d-focus]')
    );
    focusButtons.forEach((btn) => {
      btn.addEventListener('click', () => {
        if (!this.brain3dRenderer) return;
        const group = Number(btn.dataset.brain3dFocus);
        this.brain3dRenderer.setFocus(group);
        focusButtons.forEach((b) => {
          const active = b === btn;
          b.classList.toggle('is-active', active);
          b.setAttribute('aria-pressed', active ? 'true' : 'false');
        });
      });
    });

    const viewButtons = Array.from(
      this.brain3dRoot.querySelectorAll('[data-brain3d-view]')
    );
    viewButtons.forEach((btn) => {
      btn.addEventListener('click', () => {
        if (this.brain3dRenderer) {
          this.brain3dRenderer.setView(btn.dataset.brain3dView);
        }
      });
    });

    const edgesBtn = document.getElementById('btn-brain3d-edges');
    if (edgesBtn) {
      edgesBtn.addEventListener('click', () => {
        if (!this.brain3dRenderer) return;
        const next = !edgesBtn.classList.contains('is-active');
        edgesBtn.classList.toggle('is-active', next);
        edgesBtn.setAttribute('aria-pressed', next ? 'true' : 'false');
        this.brain3dRenderer.setEdgesVisible(next);
      });
    }

    const labelsBtn = document.getElementById('btn-brain3d-labels');
    if (labelsBtn) {
      labelsBtn.addEventListener('click', () => {
        if (!this.brain3dRenderer) return;
        const next = !labelsBtn.classList.contains('is-active');
        labelsBtn.classList.toggle('is-active', next);
        labelsBtn.setAttribute('aria-pressed', next ? 'true' : 'false');
        this.brain3dRenderer.setLabelsVisible(next);
      });
    }

    const rotateBtn = document.getElementById('btn-brain3d-rotate');
    if (rotateBtn) {
      rotateBtn.addEventListener('click', () => {
        if (!this.brain3dRenderer) return;
        const next = !rotateBtn.classList.contains('is-active');
        rotateBtn.classList.toggle('is-active', next);
        rotateBtn.setAttribute('aria-pressed', next ? 'true' : 'false');
        this.brain3dRenderer.setAutoRotate(next);
      });
    }

    const fsBtn = document.getElementById('btn-brain3d-fullscreen');
    if (fsBtn) {
      fsBtn.addEventListener('click', () => {
        if (this.brain3dRenderer) {
          this.brain3dRenderer.toggleFullscreen();
        }
      });
      document.addEventListener('fullscreenchange', () => {
        const isFs =
          document.fullscreenElement ===
          this.brain3dRoot.querySelector('[data-brain3d="stage"]');
        fsBtn.textContent = isFs ? 'Exit Fullscreen' : 'Fullscreen';
      });
    }

    const closeInspectorBtn = document.getElementById('btn-brain3d-inspector-close');
    if (closeInspectorBtn) {
      closeInspectorBtn.addEventListener('click', () => {
        if (this.brain3dRenderer) {
          this.brain3dRenderer.clearSelection();
        }
      });
    }
  }

  _tabFromPathname(pathname) {
    const normalized = String(pathname || '/').replace(/\/+$/, '') || '/';
    return ROUTE_TO_TAB[normalized] || 'simulation';
  }

  _activateTab(targetTabName, { updateHistory = true, replaceHistory = false } = {}) {
    const resolvedTab = TAB_ROUTES[targetTabName] ? targetTabName : 'simulation';
    const tabButtons = Array.from(document.querySelectorAll('[role="tab"]'));
    const tabPanels = Array.from(document.querySelectorAll('[role="tabpanel"]'));

    this.activeTab = resolvedTab;
    tabButtons.forEach((btn) => {
      const selected = btn.dataset.tab === resolvedTab;
      btn.classList.toggle('is-active', selected);
      btn.setAttribute('aria-selected', selected ? 'true' : 'false');
      btn.setAttribute('tabindex', selected ? '0' : '-1');
    });

    tabPanels.forEach((panel) => {
      const matches = panel.id === `panel-${resolvedTab}`;
      panel.classList.toggle('is-active', matches);
      panel.hidden = !matches;
    });

    if (TAB_TITLES[resolvedTab]) {
      document.title = TAB_TITLES[resolvedTab];
    }

    if (updateHistory && window.history && typeof window.history.pushState === 'function') {
      const targetRoute = TAB_ROUTES[resolvedTab];
      const currentPath = window.location.pathname.replace(/\/+$/, '') || '/';
      if (currentPath !== targetRoute) {
        if (replaceHistory) {
          window.history.replaceState({ tab: resolvedTab }, '', targetRoute);
        } else {
          window.history.pushState({ tab: resolvedTab }, '', targetRoute);
        }
      }
    }

    this.trainingStudio.setVisible(resolvedTab === 'training');
    if (this.architectureView) {
      this.architectureView.setVisible(resolvedTab === 'architecture');
    }
    if (this.documentationView) {
      this.documentationView.setVisible(resolvedTab === 'docs');
    }
    if (this.brain3dRenderer) {
      this.brain3dRenderer.setVisible(
        resolvedTab === 'simulation' &&
          this.connectomeViewMode === '3d' &&
          this.isFlyModelActive
      );
    }
    if (resolvedTab === 'simulation') {
      this.intersectionRenderer.renderFrame(performance.now());
    }
  }

  _initTabs() {
    const tabButtons = Array.from(document.querySelectorAll('[role="tab"]'));

    tabButtons.forEach((btn, idx) => {
      btn.addEventListener('click', (event) => {
        // Preserve standard open-in-new-tab browser shortcuts (Ctrl/Cmd/Shift/Middle click).
        if (event.metaKey || event.ctrlKey || event.shiftKey || event.altKey || event.button !== 0) {
          return;
        }
        event.preventDefault();
        this._activateTab(btn.dataset.tab, { updateHistory: true });
      });
      btn.addEventListener('keydown', (event) => {
        if (event.key === 'ArrowRight') {
          event.preventDefault();
          const next = tabButtons[(idx + 1) % tabButtons.length];
          next.focus();
          this._activateTab(next.dataset.tab, { updateHistory: true });
        } else if (event.key === 'ArrowLeft') {
          event.preventDefault();
          const prev = tabButtons[(idx - 1 + tabButtons.length) % tabButtons.length];
          prev.focus();
          this._activateTab(prev.dataset.tab, { updateHistory: true });
        }
      });
    });

    window.addEventListener('popstate', () => {
      const tab = this._tabFromPathname(window.location.pathname);
      this._activateTab(tab, { updateHistory: false });
    });
  }

  _bindSimulationControls() {
    const playPauseBtn = document.getElementById('btn-sim-play-pause');
    const stepBtn = document.getElementById('btn-sim-step');
    const resetBtn = document.getElementById('btn-sim-reset');
    const agentSelect = document.getElementById('select-agent');
    const scenarioButtons = Array.from(document.querySelectorAll('[data-scenario]'));
    const safetyBufferButtons = Array.from(document.querySelectorAll('[data-safety-buffer]'));
    const continuousButtons = Array.from(document.querySelectorAll('[data-continuous-mode]'));
    const speedSlider = document.getElementById('slider-sim-speed');

    const gExcSlider = document.getElementById('slider-g-exc');
    const gInhSlider = document.getElementById('slider-g-inh');
    const lesionCxBtn = document.getElementById('btn-lesion-cx');
    const lesionMbBtn = document.getElementById('btn-lesion-mb');
    const lesionRandomBtn = document.getElementById('btn-lesion-random');
    const resetBrainBtn = document.getElementById('btn-reset-brain');

    const zoomOutBtn = document.getElementById('btn-zoom-out');
    const zoomInBtn = document.getElementById('btn-zoom-in');
    const zoomResetBtn = document.getElementById('btn-zoom-reset');
    const zoomLevelLabel = document.getElementById('label-zoom-level');

    const spawnLaneSelect = document.getElementById('select-spawn-lane');
    const spawnBatchSelect = document.getElementById('select-spawn-batch');
    const spawnLaneBtn = document.getElementById('btn-spawn-lane');
    const spawnRandomBtn = document.getElementById('btn-spawn-random');
    const spawnSurgeBtn = document.getElementById('btn-spawn-surge');

    const syncZoomUi = (zoomLevel) => {
      if (zoomLevelLabel) {
        zoomLevelLabel.textContent = `${Math.round(zoomLevel * 100)}%`;
      }
      if (zoomOutBtn) {
        zoomOutBtn.disabled = zoomLevel <= this.intersectionRenderer.minZoom + 1e-4;
      }
      if (zoomInBtn) {
        zoomInBtn.disabled = zoomLevel >= this.intersectionRenderer.maxZoom - 1e-4;
      }
      if (zoomResetBtn) {
        zoomResetBtn.disabled = Math.abs(zoomLevel - 1.0) <= 1e-4;
      }
    };

    this.intersectionRenderer.onZoomChange = syncZoomUi;
    syncZoomUi(this.intersectionRenderer.getZoom());

    if (zoomOutBtn) {
      zoomOutBtn.addEventListener('click', () => {
        const next = Math.round((this.intersectionRenderer.getZoom() - 0.1) * 100) / 100;
        this.intersectionRenderer.setZoom(next);
      });
    }
    if (zoomInBtn) {
      zoomInBtn.addEventListener('click', () => {
        const next = Math.round((this.intersectionRenderer.getZoom() + 0.1) * 100) / 100;
        this.intersectionRenderer.setZoom(next);
      });
    }
    if (zoomResetBtn) {
      zoomResetBtn.addEventListener('click', () => {
        this.intersectionRenderer.setZoom(1.0);
      });
    }

    const getSpawnBatch = () => {
      const parsed = Math.round(Number(spawnBatchSelect ? spawnBatchSelect.value : 1));
      return Number.isFinite(parsed) && parsed > 0 ? Math.min(40, parsed) : 1;
    };

    this.intersectionRenderer.onLaneClick = (laneId) => {
      const normalizedLane = Number(laneId);
      if (spawnLaneSelect) {
        spawnLaneSelect.value = String(normalizedLane);
      }
      this._sendSimCommand({
        action: 'spawn_vehicles',
        lane_id: normalizedLane,
        count: getSpawnBatch(),
      });
    };

    if (spawnLaneBtn) {
      spawnLaneBtn.addEventListener('click', () => {
        const laneId = Number(spawnLaneSelect ? spawnLaneSelect.value : 0);
        this._sendSimCommand({
          action: 'spawn_vehicles',
          lane_id: laneId,
          count: getSpawnBatch(),
        });
      });
    }
    if (spawnRandomBtn) {
      spawnRandomBtn.addEventListener('click', () => {
        this._sendSimCommand({ action: 'spawn_random', count: 5 });
      });
    }
    if (spawnSurgeBtn) {
      spawnSurgeBtn.addEventListener('click', () => {
        this._sendSimCommand({ action: 'spawn_surge', count_per_lane: 3 });
      });
    }

    playPauseBtn.addEventListener('click', () => {
      const nextRunning = !(this.latestSimFrame && this.latestSimFrame.running);
      this._sendSimCommand({ action: 'set_running', running: nextRunning });
    });

    stepBtn.addEventListener('click', () => {
      this._sendSimCommand({ action: 'step' });
    });

    resetBtn.addEventListener('click', () => {
      this._sendSimCommand({ action: 'reset', seed: 42 });
    });

    agentSelect.addEventListener('change', () => {
      this._sendSimCommand({ action: 'set_agent', agent_key: agentSelect.value });
    });

    scenarioButtons.forEach((btn) => {
      btn.addEventListener('click', () => {
        const scenario = btn.dataset.scenario;
        scenarioButtons.forEach((b) => {
          const active = b === btn;
          b.classList.toggle('is-active', active);
          b.setAttribute('aria-pressed', active ? 'true' : 'false');
        });
        this._sendSimCommand({ action: 'set_scenario', scenario });
      });
    });

    safetyBufferButtons.forEach((btn) => {
      btn.addEventListener('click', () => {
        const allRedSteps = Number(btn.dataset.safetyBuffer);
        safetyBufferButtons.forEach((b) => {
          const active = b === btn;
          b.classList.toggle('is-active', active);
          b.setAttribute('aria-pressed', active ? 'true' : 'false');
        });
        this._sendSimCommand({ action: 'set_safety_buffer', all_red_steps: allRedSteps });
      });
    });

    continuousButtons.forEach((btn) => {
      btn.addEventListener('click', () => {
        const enabled = btn.dataset.continuousMode === '1';
        continuousButtons.forEach((b) => {
          const active = b === btn;
          b.classList.toggle('is-active', active);
          b.setAttribute('aria-pressed', active ? 'true' : 'false');
        });
        this._sendSimCommand({ action: 'set_continuous_mode', enabled });
      });
    });

    speedSlider.addEventListener('input', () => {
      const val = Number(speedSlider.value);
      document.getElementById('label-sim-speed').textContent = `${val.toFixed(1)}/s`;
      this._sendSimCommand({ action: 'set_speed', steps_per_second: val });
    });

    const emitPharmacology = () => {
      const gExc = Number(gExcSlider.value);
      const gInh = Number(gInhSlider.value);
      document.getElementById('val-g-exc').textContent = gExc.toFixed(2);
      document.getElementById('val-g-inh').textContent = gInh.toFixed(2);
      this._sendSimCommand({
        action: 'set_pharmacology',
        g_exc: gExc,
        g_inh: gInh,
      });
    };

    gExcSlider.addEventListener('input', emitPharmacology);
    gInhSlider.addEventListener('input', emitPharmacology);

    lesionCxBtn.addEventListener('click', () => {
      this._sendSimCommand({ action: 'apply_lesion', target: 'CX', fraction: 0.25 });
    });
    lesionMbBtn.addEventListener('click', () => {
      this._sendSimCommand({ action: 'apply_lesion', target: 'MB', fraction: 0.25 });
    });
    lesionRandomBtn.addEventListener('click', () => {
      this._sendSimCommand({ action: 'apply_lesion', target: 'RANDOM', fraction: 0.25 });
    });
    resetBrainBtn.addEventListener('click', () => {
      gExcSlider.value = '1.0';
      gInhSlider.value = '1.0';
      document.getElementById('val-g-exc').textContent = '1.00';
      document.getElementById('val-g-inh').textContent = '1.00';
      this._sendSimCommand({ action: 'reset_brain' });
    });
  }

  _bindStudioAndBenchmarkCallbacks() {
    this.trainingStudio.onStartRequested = async (config) => {
      const response = await fetch('/api/training/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config),
      });
      const data = await response.json();
      if (response.ok) {
        this.isTrainingActive = true;
        this.trainingStudio.applyTrainingSnapshot(data);
      } else if (data.error) {
        document.getElementById('train-status-label').textContent = data.error;
      }
    };

    this.trainingStudio.onStopRequested = async () => {
      const response = await fetch('/api/training/stop', { method: 'POST' });
      const data = await response.json();
      this.trainingStudio.applyTrainingSnapshot(data);
    };

    this.trainingStudio.onSelectModelInSim = (agentKey) => {
      this._sendSimCommand({ action: 'set_agent', agent_key: agentKey });
      this._activateTab('simulation');
    };

    const openDocsBtn = document.getElementById('btn-open-docs-from-arch');
    if (openDocsBtn) {
      openDocsBtn.addEventListener('click', () => {
        this._activateTab('docs');
        window.scrollTo({ top: 0, behavior: 'auto' });
      });
    }

    this.benchmarkView.onRunBenchmarkRequested = async (agentKeys) => {
      this.benchmarkView.setRunningState(true);
      this.benchmarkView.setError(null);
      try {
        const response = await fetch('/api/benchmark/run', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ agent_keys: agentKeys }),
        });
        const summary = await response.json();
        if (response.ok) {
          this.benchmarkView.renderBenchmarkSummary(summary);
          if (this.documentationView) {
            this.documentationView.syncBenchmarkSummary(summary);
          }
        } else if (summary && summary.error) {
          this.benchmarkView.setError(summary.error);
        }
      } catch (err) {
        this.benchmarkView.setError(`Gagal menjalankan benchmark: ${err.message}`);
      } finally {
        this.benchmarkView.setRunningState(false);
      }
    };
  }

  _syncAgentSelectorOptions(checkpointsMap, activeAgentKey) {
    if (!checkpointsMap) {
      return;
    }
    const agentSelect = document.getElementById('select-agent');
    if (!agentSelect) {
      return;
    }

    const desiredKey =
      activeAgentKey ||
      (this.latestSimFrame && this.latestSimFrame.agent_key) ||
      agentSelect.value;

    const entries = Object.values(checkpointsMap).filter(
      (info) => info && (info.exists || info.key === 'fly' || info.key === 'fly_shuffled')
    );

    agentSelect.innerHTML = '';
    for (const info of entries) {
      const option = document.createElement('option');
      option.value = info.key;
      if (info.key === 'fixed_timer') {
        option.textContent = 'Fixed Timer — rule-based (0 learned weights)';
      } else {
        const fileLabel = info.checkpoint_file || info.relative_path || '-';
        const stepsText = Number(info.trained_timesteps || 0).toLocaleString();
        option.textContent = `${info.label} — ${fileLabel} (${stepsText} steps)`;
      }
      agentSelect.appendChild(option);
    }

    if (
      desiredKey &&
      Array.from(agentSelect.options).some((opt) => opt.value === desiredKey)
    ) {
      agentSelect.value = desiredKey;
    } else if (agentSelect.options.length > 0) {
      agentSelect.selectedIndex = 0;
    }
  }

  async _fetchOverview() {
    const response = await fetch('/api/overview');
    const data = await response.json();

    if (data.overview) {
      this.checkpointsInventory = data.overview.checkpoints || null;
      const activeKey =
        data.overview.active_agent_key ||
        (data.simulation ? data.simulation.agent_key : null);
      this.lastInventoryAgentKey = activeKey;
      this._syncAgentSelectorOptions(this.checkpointsInventory, activeKey);
      this.trainingStudio.renderInventory(this.checkpointsInventory, activeKey);
      this.benchmarkView.renderOverview(data.overview, data.benchmark_summary);
    }
    if (this.documentationView) {
      this.documentationView.syncBenchmarkSummary(data.benchmark_summary || null);
    }
    if (data.simulation) {
      this._applySimulationFrame(data.simulation);
    }
    if (data.training) {
      this.isTrainingActive =
        data.training.status === 'running' || data.training.status === 'stopping';
      this.trainingStudio.applyTrainingSnapshot(data.training);
    }
  }

  _connectSimulationSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    this.simSocket = new WebSocket(`${protocol}//${window.location.host}/ws/simulation`);

    this.simSocket.onmessage = (event) => {
      const payload = JSON.parse(event.data);
      if (payload.type === 'simulation_frame') {
        this._applySimulationFrame(payload);
      } else if (payload.type === 'overview_updated' && payload.overview) {
        this.checkpointsInventory = payload.overview.checkpoints || null;
        const activeKey =
          payload.overview.active_agent_key ||
          (this.latestSimFrame ? this.latestSimFrame.agent_key : null);
        this.lastInventoryAgentKey = activeKey;
        this._syncAgentSelectorOptions(this.checkpointsInventory, activeKey);
        this.trainingStudio.renderInventory(this.checkpointsInventory, activeKey);
        this.benchmarkView.renderModelSelector(this.checkpointsInventory);
      }
    };

    this.simSocket.onclose = () => {
      setTimeout(() => this._connectSimulationSocket(), 1500);
    };
  }

  _connectTrainingSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    this.trainSocket = new WebSocket(`${protocol}//${window.location.host}/ws/training`);

    this.trainSocket.onmessage = (event) => {
      const payload = JSON.parse(event.data);
      if (payload.type === 'training_state' || payload.type === 'training_started') {
        this.isTrainingActive =
          payload.status === 'running' || payload.status === 'stopping';
        this.trainingStudio.applyTrainingSnapshot(payload);
      } else if (payload.type === 'training_update') {
        this.isTrainingActive = true;
        this.trainingStudio.appendTrainingUpdate(payload);
      } else if (
        payload.type === 'training_completed' ||
        payload.type === 'training_cancelled' ||
        payload.type === 'training_error'
      ) {
        this.isTrainingActive = false;
        this.trainingStudio.applyTrainingSnapshot(payload);
        this._fetchOverview();
      }
    };

    this.trainSocket.onclose = () => {
      setTimeout(() => this._connectTrainingSocket(), 2000);
    };
  }

  async _sendSimCommand(commandObj) {
    if (this.simSocket && this.simSocket.readyState === WebSocket.OPEN) {
      this.simSocket.send(JSON.stringify(commandObj));
      return;
    }

    const response = await fetch('/api/simulation/control', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(commandObj),
    });
    if (response.ok) {
      const frame = await response.json();
      this._applySimulationFrame(frame);
    }
  }

  _applySimulationFrame(frame) {
    this.latestSimFrame = frame;
    this.intersectionRenderer.applyBackendFrame(frame);

    const snapshot = frame.snapshot || {};
    const policy = frame.policy_telemetry || {};
    const brain = frame.connectome_telemetry || {};
    const modelTel = frame.model_telemetry || {};
    const isFlyActive = Boolean(brain.is_fly_active);
    this.isFlyModelActive = isFlyActive;

    this._syncRightColumnForModel(frame, brain, modelTel, isFlyActive);

    if (isFlyActive) {
      if (this.connectomeViewMode === '2d') {
        this.connectomeRenderer.render(brain, frame.agent_name);
      }
      if (this.brain3dRenderer && this.brain3dRenderer.ready) {
        this.brain3dRenderer.applyFrame(frame);
      }
    } else if (this.modelTelemetryRenderer) {
      this.modelTelemetryRenderer.render(frame);
    }

    // Top telemetry strip
    const isContinuous = Boolean(snapshot.continuous_mode);
    const regimeEl = document.getElementById('sim-regime-value');
    if (regimeEl) {
      regimeEl.textContent = isContinuous
        ? `${snapshot.step ?? 0}`
        : `${snapshot.step ?? 0} / ${snapshot.max_steps ?? 300}`;
    }
    const signalTextEl = document.getElementById('sim-signal-text');
    const signalDotEl = document.getElementById('sim-signal-dot');
    const stageStr = String(snapshot.signal_stage || 'GREEN').toUpperCase();
    if (signalTextEl) {
      signalTextEl.textContent = `${snapshot.phase_label || '-'} · ${stageStr}`;
    }
    if (signalDotEl) {
      signalDotEl.classList.remove('signal-dot-green', 'signal-dot-amber', 'signal-dot-red');
      if (stageStr.includes('YELLOW') || stageStr.includes('AMBER')) {
        signalDotEl.classList.add('signal-dot-amber');
      } else if (stageStr.includes('RED')) {
        signalDotEl.classList.add('signal-dot-red');
      } else {
        signalDotEl.classList.add('signal-dot-green');
      }
    }

    // Sync controls
    document.getElementById('btn-sim-play-pause').textContent = frame.running ? 'Pause' : 'Play';
    const agentSelect = document.getElementById('select-agent');
    if (
      agentSelect &&
      Array.from(agentSelect.options).some((opt) => opt.value === frame.agent_key)
    ) {
      agentSelect.value = frame.agent_key;
    }
    if (
      this.checkpointsInventory &&
      frame.agent_key &&
      frame.agent_key !== this.lastInventoryAgentKey
    ) {
      this.lastInventoryAgentKey = frame.agent_key;
      this.trainingStudio.renderInventory(this.checkpointsInventory, frame.agent_key);
    }
    document.querySelectorAll('[data-scenario]').forEach((btn) => {
      const active = btn.dataset.scenario === frame.scenario;
      btn.classList.toggle('is-active', active);
      btn.setAttribute('aria-pressed', active ? 'true' : 'false');
    });
    if (snapshot.all_red_steps !== undefined && snapshot.all_red_steps !== null) {
      const targetBuffer = Number(snapshot.all_red_steps) > 0 ? '1' : '0';
      document.querySelectorAll('[data-safety-buffer]').forEach((btn) => {
        const active = btn.dataset.safetyBuffer === targetBuffer;
        btn.classList.toggle('is-active', active);
        btn.setAttribute('aria-pressed', active ? 'true' : 'false');
      });
    }
    if (snapshot.continuous_mode !== undefined) {
      const targetContinuous = snapshot.continuous_mode ? '1' : '0';
      document.querySelectorAll('[data-continuous-mode]').forEach((btn) => {
        const active = btn.dataset.continuousMode === targetContinuous;
        btn.classList.toggle('is-active', active);
        btn.setAttribute('aria-pressed', active ? 'true' : 'false');
      });
    }
    document.getElementById('slider-sim-speed').value = String(frame.steps_per_second);
    document.getElementById('label-sim-speed').textContent = `${Number(frame.steps_per_second).toFixed(1)}/s`;

    // Metrics strip
    const totalQueue = Number(snapshot.total_queue || 0);
    const maxLaneCap = Number(snapshot.max_lane_capacity || 80);
    const maxTotalCap = Number(snapshot.max_total_capacity || maxLaneCap * 8);
    const laneQueues = Array.isArray(snapshot.lane_queues) ? snapshot.lane_queues : [];
    const maxObservedLaneQueue =
      laneQueues.length > 0 ? Math.max(...laneQueues.map((q) => Number(q || 0))) : 0;

    const capacityCapEl = document.getElementById('metric-capacity-cap');
    if (capacityCapEl) {
      capacityCapEl.textContent = `Queue Limit: ${totalQueue} / ${maxTotalCap} veh (max ${maxLaneCap}/lane)`;
      const isAtCap = maxObservedLaneQueue >= maxLaneCap || totalQueue >= maxTotalCap;
      const isNearCap =
        !isAtCap &&
        (maxObservedLaneQueue >= Math.floor(maxLaneCap * 0.75) ||
          totalQueue >= Math.floor(maxTotalCap * 0.75));
      capacityCapEl.classList.toggle('is-at-cap', isAtCap);
      capacityCapEl.classList.toggle('is-near-cap', isNearCap);
    }

    document.getElementById('metric-total-queue').textContent = `${totalQueue} veh`;
    document.getElementById('metric-mean-wait').textContent = `${Number(snapshot.mean_wait_time_s || 0).toFixed(1)} s`;
    document.getElementById('metric-max-wait').textContent = `${Number(snapshot.max_wait_time_s || 0).toFixed(1)} s`;
    document.getElementById('metric-throughput').textContent = `${snapshot.total_throughput} veh`;
    document.getElementById('metric-switches').textContent = String(snapshot.total_switches);
    document.getElementById('metric-episode-reward').textContent = Number(snapshot.episode_reward || 0).toFixed(2);
    const totalCollisions = Number(snapshot.total_collisions || 0);
    document.getElementById('metric-collisions').textContent = String(totalCollisions);
    document
      .getElementById('metric-collisions-cell')
      .classList.toggle('has-warning', totalCollisions > 0);

    // Episode crash statistics & reduction trend
    const epHistory = Array.isArray(frame.episode_history) ? frame.episode_history : [];
    this._renderEpisodeHistory(epHistory, totalCollisions);

    // Connectome readouts (only relevant when Fly / Shuffled Connectome is active)
    if (isFlyActive) {
      document.getElementById('brain-active-neurons').textContent = `${brain.active_neurons} / ${brain.total_neurons}`;
      document.getElementById('brain-tau').textContent = Number(brain.leak_rate_tau).toFixed(3);
      document.getElementById('brain-g-exc').textContent = Number(brain.g_exc).toFixed(2);
      document.getElementById('brain-g-inh').textContent = Number(brain.g_inh).toFixed(2);
      document.getElementById('brain-cx-act').textContent = Number(brain.cx_mean_abs_activity).toFixed(4);
      document.getElementById('brain-mb-act').textContent = Number(brain.mb_mean_abs_activity).toFixed(4);
      document.getElementById('brain-value-est').textContent =
        policy.value_estimate !== null && policy.value_estimate !== undefined
          ? Number(policy.value_estimate).toFixed(3)
          : '-';
    }

    // Policy probabilities
    const probs = policy.action_probabilities || [0.25, 0.25, 0.25, 0.25];
    const chosen = Number(policy.chosen_action || 0);
    document.getElementById('policy-chosen-label').textContent = `Selected: ${PHASE_NAMES[chosen] || chosen}`;

    for (let i = 0; i < 4; i += 1) {
      const ratio = Math.min(Math.max(Number(probs[i] || 0), 0), 1);
      const pct = Math.round(ratio * 100);
      const row = document.querySelector(`.prob-row[data-phase="${i}"]`);
      const fill = document.getElementById(`prob-bar-${i}`);
      const valLabel = document.getElementById(`prob-val-${i}`);
      if (row && fill && valLabel) {
        row.classList.toggle('is-selected', i === chosen);
        fill.style.transform = `scaleX(${ratio.toFixed(4)})`;
        valLabel.textContent = `${pct}%`;
      }
    }

    // Enable/disable pharmacology controls when Fly Connectome is active
    const flyOnlyIds = [
      'slider-g-exc',
      'slider-g-inh',
      'btn-lesion-cx',
      'btn-lesion-mb',
      'btn-lesion-random',
      'btn-reset-brain',
    ];
    for (const elementId of flyOnlyIds) {
      const el = document.getElementById(elementId);
      if (el) {
        el.disabled = !isFlyActive;
      }
    }
  }

  _syncRightColumnForModel(frame, brain, modelTel, isFlyActive) {
    const flyBody = document.getElementById('fly-panel-body');
    const baselineBody = document.getElementById('baseline-panel-body');
    const viewSwitch = document.getElementById('brain-view-switch');
    const titleEl = document.getElementById('right-col-title');
    const captionEl = document.getElementById('right-col-caption');
    const shuffledNotice = document.getElementById('shuffled-null-notice');
    const policyHeadingEl = document.getElementById('policy-output-heading');

    if (flyBody) {
      flyBody.hidden = !isFlyActive;
      flyBody.classList.toggle('is-hidden', !isFlyActive);
    }
    if (baselineBody) {
      baselineBody.hidden = isFlyActive;
      baselineBody.classList.toggle('is-hidden', isFlyActive);
    }
    if (viewSwitch) {
      viewSwitch.hidden = !isFlyActive;
      viewSwitch.classList.toggle('is-hidden', !isFlyActive);
    }

    if (this.brain3dRenderer) {
      this.brain3dRenderer.setVisible(
        isFlyActive && this.connectomeViewMode === '3d' && this.activeTab === 'simulation'
      );
    }

    const kind = resolveControllerKind(frame);
    const prov = frame.checkpoint_provenance || {};
    const isShuffled = Boolean(modelTel.is_shuffled || kind === 'fly_shuffled');

    if (shuffledNotice) {
      shuffledNotice.hidden = !isShuffled;
    }

    if (titleEl && captionEl) {
      if (kind === 'fly') {
        const cx = modelTel.cx_count ?? 1800;
        const mb = modelTel.mb_count ?? 700;
        const total = modelTel.num_neurons ?? brain.total_neurons ?? 2500;
        titleEl.textContent = 'Connectome Activity';
        captionEl.textContent = `FAFB v783 · ${Number(total).toLocaleString()} neuron (${Number(cx).toLocaleString()} CX, ${Number(mb).toLocaleString()} MB) · live h_t`;
      } else if (kind === 'fly_shuffled') {
        const total = modelTel.num_neurons ?? brain.total_neurons ?? 2500;
        titleEl.textContent = 'Shuffled Connectome Activity (Null Model)';
        captionEl.textContent = `${Number(total).toLocaleString()} neuron · sisi sinaptik diacak seragam (seed 42), sparsity ${((modelTel.sparsity || 0.986053) * 100).toFixed(2)}%`;
      } else if (kind === 'gru') {
        const params = Number(modelTel.trainable_params || prov.trainable_params || 69725);
        titleEl.textContent = 'GRU Recurrent State';
        captionEl.textContent = `${modelTel.hidden_dim || 105} unit GRUCell · ${params.toLocaleString()} parameter terlatih`;
      } else if (kind === 'mlp') {
        const width = modelTel.hidden_width || 250;
        const params = Number(modelTel.trainable_params || prov.trainable_params || 69755);
        titleEl.textContent = 'MLP Feedforward Activations';
        captionEl.textContent = `2 lapis Tanh (${width} + ${width} unit) · tanpa memori rekuren · ${params.toLocaleString()} parameter`;
      } else {
        titleEl.textContent = 'Fixed Timer Schedule';
        captionEl.textContent = 'Siklus aturan tetap (8, 4, 8, 4) langkah · 0 parameter terlatih';
      }
    }

    if (policyHeadingEl) {
      policyHeadingEl.textContent =
        kind === 'fixed_timer'
          ? 'Phase Selection Output (Deterministic)'
          : 'Action Policy Output (4 Phases)';
    }
  }

  _renderEpisodeHistory(epHistory, currentCollisions) {
    const summaryBadge = document.getElementById('episode-history-summary');
    const tableBody = document.getElementById('episode-history-tbody');
    const metricCrashRed = document.getElementById('metric-crash-reduction');
    const metricCrashCell = document.getElementById('metric-crash-stats-cell');

    if (epHistory.length === 0) {
      if (summaryBadge) {
        summaryBadge.textContent = 'Episode 1 in progress';
      }
      if (tableBody) {
        tableBody.innerHTML =
          '<tr><td colspan="7" class="empty-cell">Belum ada episode simulasi yang selesai. Jalankan simulasi untuk merekam log crash per episode.</td></tr>';
      }
      if (metricCrashRed) {
        metricCrashRed.textContent = `0 ep · ${currentCollisions} coll`;
      }
      if (metricCrashCell) {
        metricCrashCell.classList.remove('has-warning', 'has-success');
      }
      return;
    }

    const totalEps = epHistory.length;
    if (summaryBadge) {
      summaryBadge.textContent = `${totalEps} episodes completed`;
    }

    const baselineColl = Number(epHistory[0].total_collisions ?? 0);
    const latestEp = epHistory[epHistory.length - 1];
    const latestColl = Number(latestEp.total_collisions ?? 0);

    if (metricCrashRed) {
      if (baselineColl > 0) {
        const overallRedPct = Math.round(((baselineColl - latestColl) / baselineColl) * 100);
        const sign = overallRedPct >= 0 ? '↓' : '↑';
        metricCrashRed.textContent = `${latestColl} coll (${sign}${Math.abs(overallRedPct)}%)`;
      } else {
        metricCrashRed.textContent = `${latestColl} coll (base 0)`;
      }
    }

    if (metricCrashCell) {
      metricCrashCell.classList.toggle('has-success', baselineColl > 0 && latestColl < baselineColl);
      metricCrashCell.classList.toggle('has-warning', latestColl > 1);
    }

    if (!tableBody) {
      return;
    }

    const reversed = epHistory.slice(-20).reverse();
    tableBody.innerHTML = reversed
      .map((ep, idx) => {
        const epIndex = totalEps - idx;
        const coll = Number(ep.total_collisions ?? 0);
        let redHtml = '<span class="crash-reduction-neutral">-</span>';

        if (baselineColl > 0) {
          const redPct = Math.round(((baselineColl - coll) / baselineColl) * 100);
          if (redPct > 0) {
            redHtml = `<span class="crash-reduction-positive">↓${redPct}%</span>`;
          } else if (redPct < 0) {
            redHtml = `<span class="crash-reduction-negative">↑${Math.abs(redPct)}%</span>`;
          } else {
            redHtml = '<span class="crash-reduction-neutral">0%</span>';
          }
        } else if (coll === 0) {
          redHtml = '<span class="crash-reduction-positive">0 (Clean)</span>';
        }

        const modelLabel = String(ep.agent_key || '-').replace('ckpt:', '');
        const scenarioLabel = String(ep.scenario || '-');
        const throughput = Number(ep.total_throughput ?? 0);
        const reward = Number(ep.episode_reward ?? 0).toFixed(2);

        return `
        <tr>
          <td class="mono-value">#${epIndex}</td>
          <td>${modelLabel}</td>
          <td>${scenarioLabel}</td>
          <td class="mono-value ${coll > 0 ? 'has-warning' : ''}">${coll}</td>
          <td class="mono-value">${redHtml}</td>
          <td class="mono-value">${throughput}</td>
          <td class="mono-value">${reward}</td>
        </tr>
      `;
      })
      .join('');
  }
}

window.addEventListener('DOMContentLoaded', () => {
  const app = new TrafficLabApp();
  app.init();
});
