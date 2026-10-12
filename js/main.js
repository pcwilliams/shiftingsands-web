// Bootstrap and controller: the web version of HourglassSceneView.Coordinator.
//
// Wires the settings store, the timer, the controls and the scene together,
// and runs the frame loop.

import {
  PHYSICS, MODE_INFO, gravityScale, neckDamping, gpuSubsteps, cpuSubsteps,
  searchWorkPerFrame, formatCount,
} from './models/physicsConfig.js';
import { SettingsStore } from './services/settingsStore.js';
import { TimerModel } from './services/timerModel.js';
import { parseLaunchArgs, applyLaunchArgs } from './services/launchArgs.js';
import { summarise } from './physics/cpuSimulation.js';
import { Controls } from './ui/controls.js';
import { RunTimeline } from './models/perfHistory.js';
import { PerfChart } from './ui/perfChart.js';

/**
 * Ask for a GPU device. WebGPU hands out access in two steps: an adapter
 * describes a physical GPU, and a device is our private connection to it.
 * Returns null if any step fails, and the app carries on in CPU mode.
 */
async function requestDevice() {
  if (!navigator.gpu) return { device: null, reason: 'This browser doesn’t have WebGPU, so the sand runs in CPU mode.' };
  try {
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) return { device: null, reason: 'WebGPU is here but no GPU adapter is available, so the sand runs in CPU mode.' };
    const device = await adapter.requestDevice();
    return { device, adapter, reason: null };
  } catch (error) {
    return { device: null, reason: `WebGPU failed to start (${error.message}), so the sand runs in CPU mode.` };
  }
}

async function main() {
  const args = parseLaunchArgs(location.search);
  const store = new SettingsStore();
  applyLaunchArgs(args, store);

  const { device, adapter, reason } = await requestDevice();
  if (!device && MODE_INFO[store.mode].needsWebGPU) store.setMode('cpu', { persist: false });

  // three.js is the one big download, so it is imported only once we know
  // which renderer we need.
  const { HourglassScene } = await import('./scene/hourglassScene.js');
  const canvas = document.getElementById('scene');
  const scene = await HourglassScene.create(canvas, device);

  const timer = new TimerModel(store.duration);
  let built = null;           // the settings the current sand was built with
  let lastResults = null;
  const perf = new RunTimeline();
  const perfChart = new PerfChart(document.getElementById('perf-chart'), document.getElementById('perf-legend'));
  let gpuFrameMs = null;      // physics + drawing, as the GPU queue saw it
  let gpuFramePending = false;
  let lastGpuError = '';

  const wanted = () => ({
    mode: store.mode, count: store.count, size: store.size, randomColors: store.randomColors,
  });
  const isCurrent = () => built && Object.entries(wanted()).every(([k, v]) => built[k] === v);

  function rebuild() {
    built = wanted();
    scene.setParticles({ ...built, broadphase: store.broadphase, seed: args.seed });
    scene.engine?.resetTiming();
    perf.clear();
  }

  function start() {
    // Keep the settled sand where it lies unless a setting changed.
    if (!isCurrent()) rebuild();
    scene.engine?.resetTiming();
    // The chart's time axis is this run: flip, countdown, a little settling.
    perf.begin(performance.now() / 1000, PHYSICS.flipDuration, timer.duration);
    timer.startFlip();
    scene.startFlip(() => timer.completeFlip(performance.now() / 1000));
  }

  /** A manual reset puts fresh sand at the bottom. */
  function reset() {
    timer.reset();
    rebuild();
  }

  function startOrReset() {
    if (timer.isIdle) {
      if (timer.isComplete) timer.reset();
      start();
    } else {
      reset();
    }
  }

  const controls = new Controls({
    store, timer, hasWebGPU: Boolean(device),
    actions: {
      startOrReset,
      sleepView: (on) => scene.setSleepView(on),
    },
  });
  if (reason) controls.notify(reason);
  if (args.stats || args.test) controls.showStats(true);

  store.subscribe((_, change) => {
    switch (change) {
      case 'mode': {
        // A mode change mid-run restarts straight away in the new mode.
        const wasBusy = !timer.isIdle;
        timer.reset();
        rebuild();
        if (wasBusy) start();
        break;
      }
      case 'broadphase':
        // Same particles, same buffers, different search: switch it live.
        // The one exception: all-pairs has a lower ceiling, so if the count
        // had to drop, start again with the smaller pile.
        if (store.count !== built.count) {
          timer.reset();
          rebuild();
        } else {
          scene.setBroadphase(store.broadphase);
        }
        break;
      case 'duration':
        timer.setDuration(store.duration);
        break;
      default:
        // Count, size and colours respawn the sand, so they wait until the
        // hourglass is at rest.
        if (timer.isIdle) {
          if (timer.isComplete) timer.reset();
          rebuild();
        }
    }
  });

  timer.subscribe(() => {
    if (timer.isComplete && args.test) dumpResults();
  });

  async function dumpResults() {
    const data = await scene.readParticles();
    lastResults = { mode: scene.mode, broadphase: store.broadphase, duration: timer.duration, ...summarise(data, scene.count) };
    globalThis.shiftingSands.results = lastResults;
    console.log('[TEST]', JSON.stringify(lastResults));
  }

  if (device) {
    // WebGPU reports most mistakes asynchronously and silently: the bad call
    // returns normally and the frame just comes out wrong. Put the first one
    // on screen so a blank hourglass explains itself.
    let reported = false;
    device.addEventListener('uncapturederror', (event) => {
      console.error('WebGPU error:', event.error.message);
      lastGpuError = event.error.message;
      if (reported) return;
      reported = true;
      controls.notify(`WebGPU error: ${event.error.message.slice(0, 220)}`);
    });
    device.lost.then((info) => {
      if (info.reason !== 'destroyed') controls.notify('The GPU device was lost. Reload the page to carry on.');
    });
  }

  rebuild();

  // ── Frame loop ─────────────────────────────────────────────────────────
  let last = 0;
  let smoothedMs = 16.7;
  let frames = 0;

  function frame(nowMs) {
    requestAnimationFrame(frame);
    const rawDt = last ? (nowMs - last) / 1000 : 1 / 60;
    last = nowMs;
    // A long gap (tab in the background, a slow frame) must not turn into one
    // huge physics step, or grains would tunnel through each other.
    const dt = Math.min(rawDt, PHYSICS.maxDt);

    const beforeFrame = performance.now();
    scene.update(dt, {
      gravity: gravityScale(timer.duration),
      neckDamping: neckDamping(timer.duration),
    });
    timer.tick(nowMs / 1000);
    const beforeDraw = performance.now();
    scene.render();
    timeFrameOnGpu(scene.mode === 'cpu' ? beforeDraw : beforeFrame);
    recordPerf();

    smoothedMs += (rawDt * 1000 - smoothedMs) * 0.05;
    if (++frames % 20 === 0) controls.setStats(statsText());
    if (controls.statsVisible && frames % 4 === 0) {
      perfChart.draw(perf, {
        physicsLabel: scene.mode === 'cpu' ? 'physics (CPU)' : 'physics (GPU)',
        hasDrawing: Boolean(device),
      });
    }
  }

  // How long the GPU took over this frame's work: from when it was queued to
  // when the queue reports it done. A new measurement starts as soon as the
  // last one lands, so when the GPU keeps up every frame is timed.
  function timeFrameOnGpu(queuedAt) {
    if (!device || gpuFramePending) return;
    gpuFramePending = true;
    device.queue.onSubmittedWorkDone().then(() => {
      gpuFrameMs = performance.now() - queuedAt;
      gpuFramePending = false;
    }, () => { gpuFramePending = false; });
  }

  // One sample per frame. In the GPU modes the queue runs physics and then
  // the draw, so drawing is the whole frame less the physics.
  function recordPerf() {
    const now = performance.now() / 1000;
    if (scene.mode === 'cpu') {
      perf.push(now, scene.cpuPhysicsMs ?? 0, gpuFrameMs ?? 0);
    } else {
      const physics = scene.engine.gpuMs ?? 0;
      perf.push(now, physics, Math.max((gpuFrameMs ?? physics) - physics, 0));
    }
  }

  function statsText() {
    const mode = scene.mode;
    const substeps = mode === 'cpu' ? cpuSubsteps(scene.count) : gpuSubsteps(scene.count);
    const work = searchWorkPerFrame(mode, scene.count, store.broadphase);
    const lines = [
      `${(1000 / smoothedMs).toFixed(0)} fps  ${smoothedMs.toFixed(1)} ms`,
      `${MODE_INFO[mode].label}  ${formatCount(scene.count)} grains  r ${scene.particleRadius.toFixed(4)}`,
      `${substeps} substep${substeps > 1 ? 's' : ''}  ${work.value.toExponential(1)} ${work.unit}/frame`,
      `${(scene.triangles / 1e6).toFixed(2)}M triangles`,
    ];
    if (mode !== 'cpu') {
      lines.push(`${(scene.engine.memoryBytes / 1048576).toFixed(1)} MB of GPU buffers`);
    }
    // Plumbing, for diagnosing a blank picture.
    const backend = scene.renderer.backend?.isWebGPUBackend ? 'WebGPU' : 'WebGL 2';
    let feed = 'array upload';
    if (mode === 'direct') feed = scene.gpuVertexBuffer ? 'GPU copy bound' : 'GPU copy waiting';
    if (mode === 'gpu') feed = `${scene.engine.readsCompleted ?? 0} readbacks`;
    lines.push(`${backend}  frame ${scene.framesRendered ?? 0}  ${feed}`);
    if (lastGpuError) lines.push(`error: ${lastGpuError.slice(0, 60)}`);
    if (lastResults) {
      lines.push(`drained ${lastResults.drainedPercent.toFixed(0)}%  asleep ${lastResults.sleeping}`);
    }
    return lines.join('\n');
  }

  controls.onStatsShown = () => controls.setStats(statsText());
  if (controls.statsVisible) controls.onStatsShown();

  // For poking at from devtools, and for the headless checks.
  globalThis.shiftingSands = { scene, store, timer, start, reset, adapter, results: null };
  document.body.classList.add('ready');
  requestAnimationFrame(frame);

  if (args.test || args.autostart) setTimeout(start, 2000);
}

main().catch((error) => {
  console.error(error);
  const notice = document.getElementById('notice');
  notice.textContent = `Something went wrong starting the hourglass: ${error.message}`;
  notice.hidden = false;
});
