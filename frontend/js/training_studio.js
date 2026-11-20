/**
 * Transparent CPU PPO Training Studio view with real-time WebSocket charts,
 * Live Policy Preview (rush_hour), and per-update log table.
 */

import { IntersectionRenderer } from './intersection_renderer.js';

export class TrainingStudioView {
  constructor() {
    this.form = document.getElementById('training-form');
    this.modelSelect = document.getElementById('train-model-type');
    this.timestepsSelect = document.getElementById('train-timesteps');
    this.customTimestepsGroup = document.getElementById('custom-timesteps-group');
    this.customTimestepsInput = document.getElementById('train-custom-timesteps');
    this.learningRateInput = document.getElementById('train-lr');
    this.seedInput = document.getElementById('train-seed');
    this.gammaInput = document.getElementById('train-gamma');
    this.gaeLambdaInput = document.getElementById('train-gae-lambda');
    this.clipCoefInput = document.getElementById('train-clip-coef');
    this.entCoefInput = document.getElementById('train-ent-coef');
    this.vfCoefInput = document.getElementById('train-vf-coef');
    this.updateEpochsInput = document.getElementById('train-update-epochs');
    this.rolloutStepsSelect = document.getElementById('train-rollout-steps');
    this.chunkLenSelect = document.getElementById('train-chunk-len');
    this.numMinibatchesSelect = document.getElementById('train-num-minibatches');
    this.batchConstraintHint = document.getElementById('train-batch-constraint-hint');
    this.annealLrCheckbox = document.getElementById('train-anneal-lr');
    this.saveOnCancelCheckbox = document.getElementById('train-save-on-cancel');
    this.saveModeSelect = document.getElementById('train-save-mode');
    this.customFilenameGroup = document.getElementById('custom-filename-group');
    this.customFilenameInput = document.getElementById('train-custom-filename');
    this.savePathPreview = document.getElementById('train-save-path-preview');
    this.startBtn = document.getElementById('btn-start-training');
    this.stopBtn = document.getElementById('btn-stop-training');

    this.statusLabel = document.getElementById('train-status-label');
    this.progressBar = document.getElementById('train-progress-bar');
    this.progressDetail = document.getElementById('train-progress-detail');
    this.inventoryList = document.getElementById('checkpoint-inventory-list');
    this.logTableBody = document.getElementById('training-log-tbody');

    this.rewardCanvas = document.getElementById('train-reward-chart');
    this.lossCanvas = document.getElementById('train-loss-chart');

    this.previewCanvas = document.getElementById('training-preview-canvas');
    this.previewCaption = document.getElementById('train-preview-caption');
    this.previewRenderer = this.previewCanvas
      ? new IntersectionRenderer(this.previewCanvas, { compact: true })
      : null;

    this.history = [];
    this.previewQueue = [];
    this.isVisible = false;
    this.previewAnimationFrameId = null;
    this.lastPreviewPlayMs = 0;
    this.previewIntervalMs = 260;

    this.onStartRequested = null;
    this.onStopRequested = null;
    this.onSelectModelInSim = null;

    this._bindEvents();
    this._updateTimestepsVisibility();
    this._updateBatchConstraintHint();
    this._updateSavePathPreview();
    if (this.previewRenderer) {
      this.previewRenderer.renderFrame(performance.now());
    }
  }

  _bindEvents() {
    this.form.addEventListener('submit', (event) => {
      event.preventDefault();
      if (this.onStartRequested) {
        this.onStartRequested(this.buildTrainingConfig());
      }
    });

    if (this.timestepsSelect) {
      this.timestepsSelect.addEventListener('change', () => {
        this._updateTimestepsVisibility();
      });
    }

    const refreshBatchHint = () => this._updateBatchConstraintHint();
    [this.rolloutStepsSelect, this.chunkLenSelect, this.numMinibatchesSelect].forEach((el) => {
      if (el) {
        el.addEventListener('change', refreshBatchHint);
      }
    });

    const refreshSavePreview = () => this._updateSavePathPreview();
    [this.saveModeSelect, this.customFilenameInput, this.modelSelect].forEach((el) => {
      if (el) {
        el.addEventListener('change', refreshSavePreview);
        el.addEventListener('input', refreshSavePreview);
      }
    });

    this.stopBtn.addEventListener('click', () => {
      if (this.onStopRequested) {
        this.onStopRequested();
      }
    });
  }

  _updateTimestepsVisibility() {
    if (!this.timestepsSelect || !this.customTimestepsGroup) {
      return;
    }
    const isCustom = this.timestepsSelect.value === 'custom';
    this.customTimestepsGroup.classList.toggle('is-hidden', !isCustom);
  }

  _updateBatchConstraintHint() {
    if (!this.batchConstraintHint) {
      return;
    }
    const rolloutSteps = Number(
      this.rolloutStepsSelect ? this.rolloutStepsSelect.value : 128
    );
    const chunkLen = Number(this.chunkLenSelect ? this.chunkLenSelect.value : 16);
    const numMinibatches = Number(
      this.numMinibatchesSelect ? this.numMinibatchesSelect.value : 4
    );
    const numEnvs = 8;
    const stepsPerUpdate = numEnvs * rolloutSteps;
    const totalChunks = Math.max(1, Math.floor((rolloutSteps / chunkLen) * numEnvs));
    const chunksPerBatch = Math.max(1, Math.floor(totalChunks / numMinibatches));

    this.batchConstraintHint.textContent = `Rollout batch: ${numEnvs} envs x ${rolloutSteps} = ${stepsPerUpdate.toLocaleString()} steps/update (${totalChunks} seq chunks, ${chunksPerBatch}/minibatch)`;
  }

  buildTrainingConfig() {
    const saveMode = this.saveModeSelect ? this.saveModeSelect.value : 'new_file';
    const customFilename =
      saveMode === 'new_file' && this.customFilenameInput
        ? this.customFilenameInput.value.trim()
        : '';

    let totalTimesteps = 40960;
    if (this.timestepsSelect && this.timestepsSelect.value === 'custom') {
      const rawCustom = this.customTimestepsInput
        ? Number(this.customTimestepsInput.value)
        : 262144;
      const safeCustom = Number.isFinite(rawCustom) ? rawCustom : 262144;
      totalTimesteps = Math.min(2000000, Math.max(1024, Math.round(safeCustom)));
    } else if (this.timestepsSelect) {
      totalTimesteps = Number(this.timestepsSelect.value || 40960);
    }

    const learningRate = Number(this.learningRateInput ? this.learningRateInput.value : 0.0008);
    const seed = Math.max(0, Math.round(Number(this.seedInput ? this.seedInput.value : 42)));
    const gamma = Number(this.gammaInput ? this.gammaInput.value : 0.99);
    const gaeLambda = Number(this.gaeLambdaInput ? this.gaeLambdaInput.value : 0.95);
    const clipCoef = Number(this.clipCoefInput ? this.clipCoefInput.value : 0.2);
    const entCoef = Number(this.entCoefInput ? this.entCoefInput.value : 0.005);
    const vfCoef = Number(this.vfCoefInput ? this.vfCoefInput.value : 0.25);
    const updateEpochs = Math.min(
      10,
      Math.max(1, Math.round(Number(this.updateEpochsInput ? this.updateEpochsInput.value : 4)))
    );
    const rolloutSteps = Number(
      this.rolloutStepsSelect ? this.rolloutStepsSelect.value : 128
    );
    const sequenceChunkLen = Number(this.chunkLenSelect ? this.chunkLenSelect.value : 16);
    const numMinibatches = Number(
      this.numMinibatchesSelect ? this.numMinibatchesSelect.value : 4
    );

    return {
      model_type: this.modelSelect ? this.modelSelect.value : 'fly',
      total_timesteps: totalTimesteps,
      learning_rate: Number.isFinite(learningRate) ? learningRate : 0.0008,
      seed: Number.isFinite(seed) ? seed : 42,
      save_mode: saveMode,
      custom_filename: customFilename,
      gamma: Number.isFinite(gamma) ? gamma : 0.99,
      gae_lambda: Number.isFinite(gaeLambda) ? gaeLambda : 0.95,
      clip_coef: Number.isFinite(clipCoef) ? clipCoef : 0.2,
      ent_coef: Number.isFinite(entCoef) ? entCoef : 0.005,
      vf_coef: Number.isFinite(vfCoef) ? vfCoef : 0.25,
      update_epochs: Number.isFinite(updateEpochs) ? updateEpochs : 4,
      rollout_steps: Number.isFinite(rolloutSteps) ? rolloutSteps : 128,
      sequence_chunk_len: Number.isFinite(sequenceChunkLen) ? sequenceChunkLen : 16,
      num_minibatches: Number.isFinite(numMinibatches) ? numMinibatches : 4,
      anneal_lr: this.annealLrCheckbox ? Boolean(this.annealLrCheckbox.checked) : true,
      save_on_cancel: this.saveOnCancelCheckbox
        ? Boolean(this.saveOnCancelCheckbox.checked)
        : true,
    };
  }

  _sanitizeCustomFilename(rawName, modelType) {
    const trimmed = String(rawName || '').trim();
    if (!trimmed) {
      return `${modelType}_ppo_<timestamp>.pt`;
    }
    const withoutExt = trimmed.toLowerCase().endsWith('.pt')
      ? trimmed.slice(0, -3)
      : trimmed;
    const cleanedStem = withoutExt
      .replace(/[^a-zA-Z0-9_.-]+/g, '_')
      .replace(/\.{2,}/g, '.')
      .replace(/^[._-]+|[._-]+$/g, '');
    if (!cleanedStem) {
      return `${modelType}_ppo_<timestamp>.pt`;
    }
    return `${cleanedStem}.pt`;
  }

  _updateSavePathPreview() {
    if (!this.saveModeSelect || !this.savePathPreview) {
      return;
    }
    const saveMode = this.saveModeSelect.value || 'new_file';
    const modelType = (this.modelSelect && this.modelSelect.value) || 'fly';

    if (this.customFilenameGroup) {
      this.customFilenameGroup.classList.toggle('is-hidden', saveMode !== 'new_file');
    }

    if (saveMode === 'new_file') {
      const rawName = this.customFilenameInput ? this.customFilenameInput.value : '';
      const filename = this._sanitizeCustomFilename(rawName, modelType);
      this.savePathPreview.textContent = `Local save path: artifacts/checkpoints/${filename}`;
    } else if (saveMode === 'overwrite_default') {
      this.savePathPreview.textContent = `Local save path: artifacts/checkpoints/${modelType}_ppo.pt`;
    } else {
      this.savePathPreview.textContent = 'In-memory dry run (not saved to local disk)';
    }
  }

  renderInventory(checkpointsMap, activeAgentKey = null) {
    if (!checkpointsMap) {
      return;
    }

    this.inventoryList.innerHTML = '';

    for (const key of Object.keys(checkpointsMap)) {
      const info = checkpointsMap[key];
      if (!info) {
        continue;
      }

      const modelKey = info.key || key;
      const isActive = Boolean(activeAgentKey && modelKey === activeAgentKey);
      const isExists = Boolean(info.exists);

      const row = document.createElement('div');
      row.className = 'inventory-row';

      const titleRow = document.createElement('div');
      titleRow.className = 'inventory-row-title';

      const labelSpan = document.createElement('span');
      labelSpan.textContent = info.label || modelKey;

      const badgeSpan = document.createElement('span');
      badgeSpan.className = 'inventory-badge mono-value';
      if (isActive) {
        badgeSpan.classList.add('is-active');
        badgeSpan.textContent = 'Active in Sim';
      } else if (isExists) {
        badgeSpan.classList.add('is-ready');
        badgeSpan.textContent = 'Ready';
      } else {
        badgeSpan.textContent = 'Missing';
      }

      titleRow.appendChild(labelSpan);
      titleRow.appendChild(badgeSpan);

      const pathMeta = document.createElement('div');
      pathMeta.className = 'inventory-row-meta mono-value';
      pathMeta.textContent = info.relative_path || info.checkpoint_file || '-';

      const actionsRow = document.createElement('div');
      actionsRow.className = 'inventory-row-actions';

      const statsSpan = document.createElement('span');
      statsSpan.className = 'inventory-row-meta mono-value';
      const paramsText = info.trainable_params
        ? `${Number(info.trainable_params).toLocaleString()} params`
        : '0 params';
      const stepsText = info.trained_timesteps
        ? `${Number(info.trained_timesteps).toLocaleString()} steps`
        : modelKey === 'fixed_timer'
          ? 'Rule-based'
          : '0 steps';
      statsSpan.textContent = `${paramsText} · ${stepsText}`;
      actionsRow.appendChild(statsSpan);

      if (isExists) {
        const loadBtn = document.createElement('button');
        loadBtn.type = 'button';
        loadBtn.className = 'btn btn-secondary btn-compact';
        loadBtn.textContent = 'Load in Live Sim';
        loadBtn.addEventListener('click', () => {
          if (this.onSelectModelInSim) {
            this.onSelectModelInSim(modelKey);
          }
        });
        actionsRow.appendChild(loadBtn);
      }

      row.appendChild(titleRow);
      row.appendChild(pathMeta);
      row.appendChild(actionsRow);
      this.inventoryList.appendChild(row);
    }
  }

  setVisible(isVisible) {
    const nextVisible = Boolean(isVisible);
    if (this.isVisible === nextVisible) {
      return;
    }
    this.isVisible = nextVisible;
    if (this.isVisible) {
      this._startPreviewLoop();
    } else {
      this._stopPreviewLoop();
    }
  }

  _startPreviewLoop() {
    if (this.previewAnimationFrameId !== null || !this.previewRenderer) {
      return;
    }
    const tick = (timestampMs) => {
      if (!this.isVisible) {
        this.previewAnimationFrameId = null;
        return;
      }
      this._advancePreviewQueue(timestampMs);
      this.previewRenderer.renderFrame(timestampMs);
      this.previewAnimationFrameId = requestAnimationFrame(tick);
    };
    this.previewAnimationFrameId = requestAnimationFrame(tick);
  }

  _stopPreviewLoop() {
    if (this.previewAnimationFrameId !== null) {
      cancelAnimationFrame(this.previewAnimationFrameId);
      this.previewAnimationFrameId = null;
    }
  }

  _enqueuePreviewFrames(updateNumber, frames) {
    if (!Array.isArray(frames) || frames.length === 0) {
      return;
    }
    const items = frames
      .filter((snapshot) => snapshot && typeof snapshot === 'object')
      .map((snapshot) => ({ updateNumber, snapshot }));

    if (items.length === 0) {
      return;
    }

    this.previewQueue = items;
    this.lastPreviewPlayMs = 0;

    if (!this.isVisible) {
      const latestItem = items[items.length - 1];
      this._updatePreviewCaption(latestItem.updateNumber, latestItem.snapshot);
    }
  }

  _advancePreviewQueue(timestampMs) {
    if (!this.previewRenderer || this.previewQueue.length === 0) {
      return;
    }
    if (
      this.lastPreviewPlayMs === 0 ||
      timestampMs - this.lastPreviewPlayMs >= this.previewIntervalMs
    ) {
      const nextItem = this.previewQueue.shift();
      this.lastPreviewPlayMs = timestampMs;
      this.previewRenderer.applyBackendFrame({
        snapshot: nextItem.snapshot,
        steps_per_second: 1000 / this.previewIntervalMs,
      });
      this._updatePreviewCaption(nextItem.updateNumber, nextItem.snapshot);
    }
  }

  _updatePreviewCaption(updateNumber, snapshot) {
    if (!this.previewCaption || !snapshot) {
      return;
    }
    const stepText = `${Number(snapshot.step || 0)}/${Number(snapshot.max_steps || 300)}`;
    const phaseText = snapshot.phase_label
      ? `${snapshot.phase_label} (${snapshot.signal_stage || 'GREEN'})`
      : '-';
    const queueText = `${Number(snapshot.total_queue || 0)} veh`;
    const collisionsCount = Number(snapshot.total_collisions || 0);

    this.previewCaption.textContent =
      `Update ${updateNumber} · Step ${stepText} · Phase ${phaseText} · Queue ${queueText} · Collisions ${collisionsCount}`;
    this.previewCaption.classList.toggle('has-warning', collisionsCount > 0);
  }

  applyTrainingSnapshot(statePayload) {
    const status = statePayload.status || 'idle';
    const isRunning = status === 'running' || status === 'stopping';

    this.statusLabel.textContent =
      status === 'cancelled' && statePayload.saved_checkpoint_file
        ? 'cancelled (saved)'
        : status;
    this.startBtn.disabled = isRunning;
    this.stopBtn.disabled = !isRunning;

    if (Array.isArray(statePayload.history)) {
      this.history = statePayload.history.slice();
    }

    const latest =
      statePayload.latest_metrics ||
      (this.history.length > 0 ? this.history[this.history.length - 1] : null);

    if (this.history.length === 0 && isRunning) {
      this.progressBar.style.transform = 'scaleX(0)';
      this.progressDetail.textContent = '0 / 0 updates (0%)';
      this.previewQueue = [];
      this.lastPreviewPlayMs = 0;
      if (this.previewCaption) {
        this.previewCaption.textContent =
          'Update 0 · Step - · Phase - · Queue - · Collisions -';
        this.previewCaption.classList.remove('has-warning');
      }
    }

    if (latest) {
      this._updateTelemetryFields(latest);
    }

    if ((status === 'completed' || status === 'cancelled') && statePayload.saved_checkpoint_file) {
      const savedFile = String(statePayload.saved_checkpoint_file);
      const savedPath = savedFile.startsWith('artifacts/')
        ? savedFile
        : `artifacts/checkpoints/${savedFile}`;
      const prefix =
        status === 'cancelled'
          ? 'Stopped & saved partial checkpoint'
          : 'Saved locally';
      const savedNote = `${prefix}: ${savedPath}`;
      const currentText = this.progressDetail.textContent.trim();
      if (!currentText.includes(savedPath)) {
        this.progressDetail.textContent = currentText
          ? `${currentText} · ${savedNote}`
          : savedNote;
      }
    } else if (status === 'cancelled') {
      const cancelNote = 'Training stopped (no checkpoint saved)';
      const currentText = this.progressDetail.textContent.trim();
      if (!currentText.includes(cancelNote)) {
        this.progressDetail.textContent = currentText
          ? `${currentText} · ${cancelNote}`
          : cancelNote;
      }
    } else if (status === 'error' && statePayload.error_message) {
      this.progressDetail.textContent = `Error: ${statePayload.error_message}`;
    }

    if (Array.isArray(statePayload.preview_frames) && statePayload.preview_frames.length > 0) {
      const updateNum = latest ? Math.round(latest.update || 0) : 0;
      this._enqueuePreviewFrames(updateNum, statePayload.preview_frames);
    }

    this._redrawCharts();
    this._rebuildLogTable();
  }

  appendTrainingUpdate(updatePayload) {
    this.statusLabel.textContent = updatePayload.status || 'running';
    this.startBtn.disabled = true;
    this.stopBtn.disabled = false;

    const metrics = updatePayload.metrics;
    const updateNum = metrics ? Math.round(metrics.update || 0) : 0;

    if (Array.isArray(updatePayload.preview_frames) && updatePayload.preview_frames.length > 0) {
      this._enqueuePreviewFrames(updateNum, updatePayload.preview_frames);
    }

    if (!metrics) {
      return;
    }

    this.history.push(metrics);
    this._updateTelemetryFields(metrics);
    this._redrawCharts();
    this._rebuildLogTable();
  }

  _updateTelemetryFields(m) {
    const updateNum = Math.round(m.update || 0);
    const totalUpdates = Math.max(1, Math.round(m.total_updates || 1));
    const ratio = Math.min(Math.max(Number(m.progress_ratio || 0), 0), 1);
    const pct = Math.min(100, Math.round(ratio * 100));

    this.progressBar.style.transform = `scaleX(${ratio.toFixed(4)})`;
    this.progressDetail.textContent = `${updateNum} / ${totalUpdates} updates (${pct}%) · ${Math.round(m.timesteps || 0).toLocaleString()} steps`;

    document.getElementById('tmetric-reward').textContent = (m.mean_episode_reward ?? 0).toFixed(2);
    document.getElementById('tmetric-throughput').textContent = `${(m.mean_throughput ?? 0).toFixed(1)} veh`;
    document.getElementById('tmetric-pg-loss').textContent = (m.policy_loss ?? 0).toFixed(4);
    document.getElementById('tmetric-v-loss').textContent = (m.value_loss ?? 0).toFixed(4);
    document.getElementById('tmetric-entropy').textContent = (m.entropy ?? 0).toFixed(3);

    const tauPart = m.leak_rate_tau && m.leak_rate_tau > 0 ? ` · tau=${m.leak_rate_tau.toFixed(3)}` : '';
    document.getElementById('tmetric-sps-tau').textContent = `${Math.round(m.steps_per_second || 0)} sps${tauPart}`;
  }

  _rebuildLogTable() {
    if (this.history.length === 0) {
      this.logTableBody.innerHTML =
        '<tr><td colspan="8" class="empty-cell">Belum ada sesi training aktif pada tab ini. Pilih konfigurasi lalu klik Start Training.</td></tr>';
      return;
    }

    const recent = this.history.slice(-15).reverse();
    this.logTableBody.innerHTML = recent
      .map(
        (m) => `
      <tr>
        <td class="mono-value">${Math.round(m.update)}/${Math.round(m.total_updates)}</td>
        <td class="mono-value">${Math.round(m.timesteps).toLocaleString()}</td>
        <td class="mono-value">${Math.round(m.steps_per_second)}</td>
        <td class="mono-value">${m.mean_episode_reward.toFixed(2)}</td>
        <td class="mono-value">${m.mean_throughput.toFixed(1)}</td>
        <td class="mono-value">${m.policy_loss.toFixed(4)}</td>
        <td class="mono-value">${m.value_loss.toFixed(4)}</td>
        <td class="mono-value">${m.entropy.toFixed(3)}</td>
      </tr>
    `
      )
      .join('');
  }

  _redrawCharts() {
    this._drawLineChart(
      this.rewardCanvas,
      [
        {
          label: 'mean_episode_reward',
          color: '#FF9500', // Apple System Orange
          values: this.history.map((m) => m.mean_episode_reward),
        },
      ],
      'timesteps'
    );

    this._drawLineChart(
      this.lossCanvas,
      [
        {
          label: 'value_loss',
          color: '#007AFF', // Apple System Blue
          values: this.history.map((m) => m.value_loss),
        },
        {
          label: 'entropy',
          color: '#34C759', // Apple System Green
          values: this.history.map((m) => m.entropy),
        },
      ],
      'timesteps'
    );
  }

  _drawLineChart(canvas, seriesList, xAxisLabel) {
    const ctx = canvas.getContext('2d');
    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, w, h);

    const steps = this.history.map((m) => m.timesteps);
    if (steps.length === 0) {
      ctx.fillStyle = '#6b7280';
      ctx.font = '400 12px "IBM Plex Sans", sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('Menunggu update training...', w / 2, h / 2);
      ctx.textAlign = 'left';
      return;
    }

    const pad = { left: 52, right: 16, top: 24, bottom: 30 };
    const plotW = w - pad.left - pad.right;
    const plotH = h - pad.top - pad.bottom;

    const allVals = seriesList.flatMap((s) => s.values);
    let minY = Math.min(...allVals);
    let maxY = Math.max(...allVals);
    if (Math.abs(maxY - minY) < 1e-6) {
      minY -= 1;
      maxY += 1;
    }

    const minX = steps[0];
    const maxX = Math.max(steps[steps.length - 1], minX + 1);

    // Grid & Y labels
    ctx.strokeStyle = '#e5e7eb';
    ctx.lineWidth = 1;
    ctx.fillStyle = '#52525b';
    ctx.font = '400 10px "IBM Plex Mono", monospace';
    for (let g = 0; g <= 4; g += 1) {
      const y = pad.top + (g / 4) * plotH;
      const val = maxY - (g / 4) * (maxY - minY);
      ctx.beginPath();
      ctx.moveTo(pad.left, y);
      ctx.lineTo(pad.left + plotW, y);
      ctx.stroke();
      ctx.fillText(val.toFixed(2), 4, y + 3);
    }

    // Plot series
    seriesList.forEach((series, idx) => {
      ctx.strokeStyle = series.color;
      ctx.lineWidth = 2;
      ctx.beginPath();
      series.values.forEach((val, i) => {
        const px = pad.left + ((steps[i] - minX) / (maxX - minX)) * plotW;
        const py = pad.top + ((maxY - val) / (maxY - minY)) * plotH;
        if (i === 0) {
          ctx.moveTo(px, py);
        } else {
          ctx.lineTo(px, py);
        }
      });
      ctx.stroke();

      // Legend badge
      ctx.fillStyle = series.color;
      if (ctx.roundRect) {
        ctx.beginPath();
        ctx.roundRect(pad.left + idx * 130, 6, 10, 10, 2);
        ctx.fill();
      } else {
        ctx.fillRect(pad.left + idx * 130, 6, 10, 10);
      }
      ctx.fillStyle = '#1d1d1f';
      ctx.font = '500 11px "IBM Plex Mono", monospace';
      ctx.fillText(series.label, pad.left + idx * 130 + 14, 15);
    });

    ctx.fillStyle = '#52525b';
    ctx.font = '400 10px "IBM Plex Mono", monospace';
    ctx.fillText(`${xAxisLabel}: ${Math.round(minX)} .. ${Math.round(maxX)}`, pad.left, h - 8);
  }
}
