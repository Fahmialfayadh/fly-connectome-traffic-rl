/**
 * Live 50x50 Drosophila sub-connectome neuron activation heatmap renderer.
 * Decodes quantized uint8 activation vectors from WebSocket frames and renders
 * CX (rows 0-35) vs MB (rows 36-49) neural states cleanly without smoothing blur.
 */

export class ConnectomeRenderer {
  constructor(canvasElement) {
    this.canvas = canvasElement;
    this.ctx = canvasElement.getContext('2d');
    this.gridSide = 50;

    this.offscreenCanvas = document.createElement('canvas');
    this.offscreenCanvas.width = this.gridSide;
    this.offscreenCanvas.height = this.gridSide;
    this.offscreenCtx = this.offscreenCanvas.getContext('2d');
    this.imageData = this.offscreenCtx.createImageData(this.gridSide, this.gridSide);
  }

  render(connectomeTelemetry, agentName) {
    const ctx = this.ctx;
    const width = this.canvas.width;
    const height = this.canvas.height;

    ctx.clearRect(0, 0, width, height);

    if (
      !connectomeTelemetry ||
      !connectomeTelemetry.is_fly_active ||
      !connectomeTelemetry.activations_b64
    ) {
      ctx.fillStyle = '#f4f4f6';
      ctx.fillRect(0, 0, width, height);
      ctx.strokeStyle = '#d4d4d8';
      ctx.strokeRect(0.5, 0.5, width - 1, height - 1);

      ctx.fillStyle = '#1f2937';
      ctx.font = '600 14px "IBM Plex Sans", sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText(`Active Controller: ${agentName || 'Baseline'}`, width / 2, height / 2 - 12);
      ctx.fillStyle = '#52525b';
      ctx.font = '400 12px "IBM Plex Sans", sans-serif';
      ctx.fillText('No 2,500-neuron connectome state for this baseline.', width / 2, height / 2 + 12);
      ctx.textAlign = 'left';
      return;
    }

    const activations = this._decodeBase64Bytes(connectomeTelemetry.activations_b64);
    const lesionMask = this._decodeBase64Bytes(connectomeTelemetry.lesion_mask_b64);
    const pixels = this.imageData.data;

    for (let idx = 0; idx < 2500; idx += 1) {
      const pixelOffset = idx * 4;
      const isAlive = lesionMask[idx] > 0;

      if (!isAlive) {
        pixels[pixelOffset] = 16;
        pixels[pixelOffset + 1] = 18;
        pixels[pixelOffset + 2] = 24;
        pixels[pixelOffset + 3] = 255;
        continue;
      }

      // Convert quantized 0..255 back to [-1, +1] diverging range
      const normalized = (activations[idx] / 255) * 2 - 1;
      const [r, g, b] = this._divergingRgb(normalized);
      pixels[pixelOffset] = r;
      pixels[pixelOffset + 1] = g;
      pixels[pixelOffset + 2] = b;
      pixels[pixelOffset + 3] = 255;
    }

    this.offscreenCtx.putImageData(this.imageData, 0, 0);
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(this.offscreenCanvas, 0, 0, width, height);

    // Anatomical band divider between CX (rows 0..35) and MB (rows 36..49)
    const splitY = Math.round((36 / 50) * height);
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.85)';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(0, splitY);
    ctx.lineTo(width, splitY);
    ctx.stroke();

    ctx.fillStyle = 'rgba(15, 23, 42, 0.78)';
    if (ctx.roundRect) {
      ctx.beginPath();
      ctx.roundRect(6, 6, 126, 20, 4);
      ctx.fill();
      ctx.beginPath();
      ctx.roundRect(6, splitY + 6, 126, 20, 4);
      ctx.fill();
    } else {
      ctx.fillRect(6, 6, 126, 20);
      ctx.fillRect(6, splitY + 6, 126, 20);
    }

    ctx.fillStyle = '#f8fafc';
    ctx.font = '600 10px "IBM Plex Mono", monospace';
    ctx.fillText('CX: PB/EB/FB/NO', 12, 19);
    ctx.fillText('MB: CA/PED/ML/VL', 12, splitY + 19);
  }

  _divergingRgb(value) {
    const base = [30, 36, 48];        // Dark graphite neutral
    const positive = [235, 118, 46];   // Warm vermilion / amber (ACh excitatory)
    const negative = [37, 99, 235];    // Cool cobalt blue (GABA/Glu inhibitory)
    const t = Math.min(Math.abs(value), 1);
    const target = value >= 0 ? positive : negative;

    return [
      Math.round(base[0] + t * (target[0] - base[0])),
      Math.round(base[1] + t * (target[1] - base[1])),
      Math.round(base[2] + t * (target[2] - base[2])),
    ];
  }

  _decodeBase64Bytes(b64) {
    const binary = atob(b64);
    const length = binary.length;
    const output = new Uint8Array(length);
    for (let i = 0; i < length; i += 1) {
      output[i] = binary.charCodeAt(i);
    }
    return output;
  }
}
