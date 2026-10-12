// The timing chart at the bottom of Stats: physics and drawing time per
// frame, in milliseconds, for one whole run. The time axis is fixed to the
// run's length and fills in from the left. One (square-root) value axis, two
// thin lines, and a dashed line at the 60 Hz frame budget. The numbers are in
// the legend underneath, so the chart never depends on colour alone.

import { niceCeiling, sqrtPosition } from '../models/perfHistory.js';

const COLORS = { physics: '#d95926', drawing: '#3987e5' };
const GRID = 'rgba(255, 255, 255, 0.16)';
const LABEL = 'rgba(255, 255, 255, 0.5)';
const BUDGET_MS = 1000 / 60;
const TOP_PAD = 12;      // room for the axis-top label
const BOTTOM_PAD = 12;   // room for the time labels

export class PerfChart {
  constructor(canvas, legend) {
    this.canvas = canvas;
    this.legend = legend;
    this.context = canvas.getContext('2d');
  }

  draw(timeline, { physicsLabel = 'physics', hasDrawing = true } = {}) {
    const { canvas, context: c } = this;
    const ratio = Math.min(globalThis.devicePixelRatio || 1, 3);
    const width = canvas.clientWidth;
    const height = canvas.clientHeight;
    if (!width || !height) return;
    if (canvas.width !== Math.round(width * ratio) || canvas.height !== Math.round(height * ratio)) {
      canvas.width = Math.round(width * ratio);
      canvas.height = Math.round(height * ratio);
    }
    c.setTransform(ratio, 0, 0, ratio, 0, 0);
    c.clearRect(0, 0, width, height);
    c.font = '10px ui-monospace, Menlo, monospace';

    const plotBottom = height - BOTTOM_PAD;
    const plotHeight = plotBottom - TOP_PAD;
    const top = niceCeiling(Math.max(timeline.maxPhysics, hasDrawing ? timeline.maxDrawing : 0));
    const y = (ms) => plotBottom - sqrtPosition(ms, top) * plotHeight;
    const x = (seconds) => (timeline.span > 0 ? (seconds / timeline.span) * width : 0);

    // Frame: baseline, axis top, the frame budget. Quiet, so data stays in front.
    c.strokeStyle = GRID;
    c.fillStyle = LABEL;
    c.lineWidth = 1;
    c.beginPath();
    c.moveTo(0, plotBottom + 0.5);
    c.lineTo(width, plotBottom + 0.5);
    c.stroke();
    c.textBaseline = 'top';
    c.textAlign = 'left';
    c.fillText(`${top} ms`, 0, 0);
    c.setLineDash([3, 3]);
    c.beginPath();
    c.moveTo(0, Math.round(y(BUDGET_MS)) + 0.5);
    c.lineTo(width, Math.round(y(BUDGET_MS)) + 0.5);
    c.stroke();
    c.setLineDash([]);
    c.textAlign = 'right';
    c.textBaseline = 'bottom';
    c.fillText('16.7', width, y(BUDGET_MS) - 1);

    if (timeline.span <= 0) {
      c.textAlign = 'center';
      c.textBaseline = 'middle';
      c.fillText('press Start to record a run', width / 2, TOP_PAD + plotHeight / 2 + 8);
    } else {
      // Time axis: its length, and a tick where the flip lands and where the
      // countdown reaches zero.
      c.textBaseline = 'top';
      c.textAlign = 'right';
      c.fillText(`${timeline.span.toFixed(1)} s`, width, plotBottom + 2);
      for (const mark of timeline.marks) {
        const px = Math.round(x(mark.t)) + 0.5;
        c.beginPath();
        c.moveTo(px, TOP_PAD);
        c.lineTo(px, plotBottom);
        c.stroke();
        c.textAlign = 'center';
        c.fillText(mark.label, Math.min(Math.max(px, 12), width - 56), plotBottom + 2);
      }

      // One point per pixel column (the worst frame in it), 1px lines.
      const columns = Math.max(1, Math.floor(width));
      for (const name of hasDrawing ? ['drawing', 'physics'] : ['physics']) {
        const values = timeline.columns(name, columns);
        c.strokeStyle = COLORS[name];
        c.lineWidth = 1;
        c.lineJoin = 'round';
        c.beginPath();
        let drawing = false;
        values.forEach((v, i) => {
          if (v === null) return;
          const px = ((i + 0.5) / columns) * width;
          if (drawing) c.lineTo(px, y(v)); else c.moveTo(px, y(v));
          drawing = true;
        });
        c.stroke();
      }
    }

    const now = timeline.latest;
    const row = (name, label, value, max) =>
      `<span class="perf-key"><i style="background:${COLORS[name]}"></i>${label} ${value.toFixed(1)} ms` +
      `<span class="perf-max">max ${max.toFixed(1)}</span></span>`;
    this.legend.innerHTML =
      row('physics', physicsLabel, now.physics, timeline.maxPhysics) +
      (hasDrawing ? row('drawing', 'drawing', now.drawing, timeline.maxDrawing) : '');
  }
}
