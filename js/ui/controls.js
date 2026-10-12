// DOM controls: a port of ContentView.swift's overlay.
//
// This module only moves values between the DOM and the store/timer. What a
// change means (rebuild the sand, restart the flip) is decided in main.js.

import {
  MODE_INFO, SIZE, sliderToCount, countToSlider, formatCount,
} from '../models/physicsConfig.js';

const $ = (id) => document.getElementById(id);

function bindSegmented(element, onPick) {
  const buttons = [...element.querySelectorAll('button')];
  for (const button of buttons) {
    button.addEventListener('click', () => { if (!button.disabled) onPick(button.dataset.value); });
  }
  // Arrow keys move between options, as a native radio group would.
  element.addEventListener('keydown', (event) => {
    const step = { ArrowRight: 1, ArrowDown: 1, ArrowLeft: -1, ArrowUp: -1 }[event.key];
    if (!step) return;
    const enabled = buttons.filter((b) => !b.disabled);
    const index = enabled.indexOf(document.activeElement);
    const next = enabled[(index + step + enabled.length) % enabled.length];
    next?.focus();
    next?.click();
    event.preventDefault();
  });
  return (value) => {
    for (const button of buttons) {
      const on = button.dataset.value === value;
      button.setAttribute('aria-checked', String(on));
      button.tabIndex = on ? 0 : -1;
    }
  };
}

function durationLabel(seconds) {
  const minutes = Math.floor(seconds / 60);
  const rest = seconds % 60;
  if (minutes > 0 && rest > 0) return `${minutes}m ${rest}s`;
  if (minutes > 0) return `${minutes} min`;
  return `${rest}s`;
}

export class Controls {
  /**
   * @param {object} deps
   * @param {SettingsStore} deps.store
   * @param {TimerModel} deps.timer
   * @param {boolean} deps.hasWebGPU
   * @param {object} deps.actions  { startOrReset, sleepView(on), stats(on) }
   */
  constructor({ store, timer, hasWebGPU, actions }) {
    this.store = store;
    this.timer = timer;
    this.hasWebGPU = hasWebGPU;

    this.setMode = bindSegmented($('mode'), (mode) => store.setMode(mode));
    this.setBroadphase = bindSegmented($('broadphase'), (bp) => store.setBroadphase(bp));
    if (!hasWebGPU) {
      for (const button of $('mode').querySelectorAll('button')) {
        if (MODE_INFO[button.dataset.value].needsWebGPU) button.disabled = true;
      }
    }

    // Sliders show the value live while dragging and commit on release, so a
    // drag through 200,000 doesn't respawn the sand at every step on the way.
    const count = $('count');
    count.addEventListener('input', () => {
      $('count-label').textContent =
        `${formatCount(sliderToCount(Number(count.value), store.mode, store.broadphase))} particles`;
    });
    count.addEventListener('change', () =>
      store.setCount(sliderToCount(Number(count.value), store.mode, store.broadphase)));

    const size = $('size');
    size.min = SIZE.min; size.max = SIZE.max; size.step = SIZE.step;
    size.addEventListener('input', () => {
      $('size-label').textContent = `Size: ${Number(size.value).toFixed(2)}x`;
    });
    size.addEventListener('change', () => store.setSize(Number(size.value)));

    const duration = $('duration');
    duration.addEventListener('input', () => store.setDuration(Number(duration.value)));
    for (const button of $('presets').querySelectorAll('button')) {
      button.addEventListener('click', () => store.setDuration(Number(button.dataset.seconds)));
    }

    $('colors').addEventListener('change', (e) => store.setRandomColors(e.target.checked));
    $('sleep-view').addEventListener('change', (e) => actions.sleepView(e.target.checked));
    $('stats-toggle').addEventListener('change', (e) => this.showStats(e.target.checked));
    $('start').addEventListener('click', () => actions.startOrReset());

    // Space or Enter anywhere (except on a control that uses them) starts or resets.
    document.addEventListener('keydown', (event) => {
      if (event.key !== ' ' && event.key !== 'Enter') return;
      if (event.target instanceof HTMLElement && event.target.closest('button, input, a')) return;
      event.preventDefault();
      actions.startOrReset();
    });

    // A browser may restore these boxes as ticked after a reload without
    // firing a change event. Start from what the boxes actually say.
    this.showStats($('stats-toggle').checked);
    actions.sleepView($('sleep-view').checked);

    store.subscribe(() => this.syncSettings());
    timer.subscribe(() => this.syncTimer());
    this.syncSettings();
    this.syncTimer();
  }

  syncSettings() {
    const { store } = this;
    this.setMode(store.mode);
    this.setBroadphase(store.broadphase);
    // Collision search only exists as a choice on the GPU.
    $('broadphase').hidden = store.mode === 'cpu';
    $('sleep-view-row').hidden = store.mode === 'cpu';

    $('count').value = countToSlider(store.count, store.mode, store.broadphase);
    $('count-label').textContent = `${formatCount(store.count)} particles`;
    $('size').value = store.size;
    $('size-label').textContent = `Size: ${store.size.toFixed(2)}x`;
    $('duration').value = store.duration;
    $('duration-label').textContent = durationLabel(store.duration);
    $('colors').checked = store.randomColors;
    for (const button of $('presets').querySelectorAll('button')) {
      button.setAttribute('aria-pressed', String(Number(button.dataset.seconds) === store.duration));
    }
  }

  syncTimer() {
    const { timer } = this;
    $('readout').textContent = timer.remainingText;
    $('start').textContent = timer.buttonLabel;
    const busy = !timer.isIdle;
    $('duration').disabled = busy;
    for (const button of $('presets').querySelectorAll('button')) button.disabled = busy;
  }

  showStats(on) {
    $('stats-toggle').checked = on;
    $('stats-panel').hidden = !on;
    this.statsVisible = on;
    if (on) this.onStatsShown?.();
  }

  setStats(text) {
    if (this.statsVisible) $('stats').textContent = text;
  }

  notify(message) {
    const notice = $('notice');
    notice.textContent = message;
    notice.hidden = !message;
  }
}
