// The little timing chart at the bottom of Stats: physics and drawing time
// per frame, in milliseconds, over the last few seconds. One axis, two lines,
// and a dashed line at the 60 Hz frame budget. The numbers live in the legend
// underneath, so the chart never depends on colour alone.

import { niceCeiling } from '../models/perfHistory.js';

const COLORS = { physics: '#d95926', drawing: '#3987e5' };
const GRID = 'rgba(255, 255, 255, 0.16)';
const LABEL = 'rgba(255, 255, 255, 0.5)';
const BUDGET_MS = 1000 / 60;

export class PerfChart {
  constructor(canvas, legend) {
    this.canvas = canvas;
    this.legend = legend;
    this.context = canvas.getContext('2d');
  }

  draw(history, { physicsLabel = 'physics', hasDrawing = true } = {}) {
    const { canvas, context: c } = this;
    const ratio = Math.min(globalThis.devicePixelRatio || 1, 2);
    const width = canvas.clientWidth;
    const height = canvas.clientHeight;
    if (!width || !height) return;
    if (canvas.width !== Math.round(width * ratio) || canvas.height !== Math.round(height * ratio)) {
      canvas.width = Math.round(width * ratio);
      canvas.height = Math.round(height * ratio);
    }
    c.setTransform(ratio, 0, 0, ratio, 0, 0);
    c.clearRect(0, 0, width, height);

    const top = niceCeiling(history.windowMax);
    const y = (ms) => height - 1 - (Math.min(ms, top) / top) * (height - 12);

    // Axis top and the frame budget: quiet, so the data stays in front.
    c.font = '10px ui-monospace, Menlo, monospace';
    c.fillStyle = LABEL;
    c.textBaseline = 'top';
    c.fillText(`${top} ms`, 2, 0);
    c.strokeStyle = GRID;
    c.lineWidth = 1;
    c.beginPath();
    c.moveTo(0, height - 0.5);
    c.lineTo(width, height - 0.5);
    c.stroke();
    if (BUDGET_MS < top) {
      c.setLineDash([3, 3]);
      c.beginPath();
      c.moveTo(0, Math.round(y(BUDGET_MS)) + 0.5);
      c.lineTo(width, Math.round(y(BUDGET_MS)) + 0.5);
      c.stroke();
      c.setLineDash([]);
      c.textBaseline = 'bottom';
      c.textAlign = 'right';
      c.fillText('16.7', width - 2, y(BUDGET_MS) - 1);
      c.textAlign = 'left';
    }

    const names = hasDrawing ? ['drawing', 'physics'] : ['physics'];
    for (const name of names) {
      const values = history.series(name);
      if (values.length < 2) continue;
      const stepX = width / (history.capacity - 1);
      const offset = (history.capacity - values.length) * stepX;   // newest at the right edge
      c.strokeStyle = COLORS[name];
      c.lineWidth = 2;
      c.lineJoin = 'round';
      c.beginPath();
      values.forEach((v, i) => {
        const px = offset + i * stepX;
        if (i === 0) c.moveTo(px, y(v)); else c.lineTo(px, y(v));
      });
      c.stroke();
    }

    const now = history.latest;
    const row = (name, label, value, max) =>
      `<span class="perf-key"><i style="background:${COLORS[name]}"></i>${label} ${value.toFixed(1)} ms` +
      `<span class="perf-max">max ${max.toFixed(1)}</span></span>`;
    this.legend.innerHTML =
      row('physics', physicsLabel, now.physics, history.maxPhysics) +
      (hasDrawing ? row('drawing', 'drawing', now.drawing, history.maxDrawing) : '');
  }
}
