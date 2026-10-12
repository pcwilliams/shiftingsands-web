// WebGPU physics engine: a port of MetalPhysicsEngine.swift.
//
// This file talks to WebGPU directly, with no three.js, so every step of
// running a compute shader is visible:
//
//   device.createShaderModule   compile WGSL text
//   device.createComputePipeline  pick an entry point in it
//   device.createBuffer         allocate GPU memory, saying how it will be used
//   device.createBindGroup      connect buffers to the shader's @binding slots
//   encoder.beginComputePass    record "run this pipeline over N threads"
//   device.queue.submit         hand the recorded commands to the GPU
//
// Nothing here waits for the GPU. Metal's waitUntilCompleted has no WebGPU
// equivalent: commands are queued and run in order, and the only way to see a
// result in JavaScript is the asynchronous readback at the bottom.

import { CHAMBER_BOTTOM, CHAMBER_TOP } from '../models/hourglassProfile.js';
import { STRIDE } from '../models/packing.js';
import { PHYSICS, gpuSubsteps } from '../models/physicsConfig.js';
import { computeGrid, scanBlocks } from '../models/grid.js';
import {
  WORKGROUP_SIZE, UNIFORM_BYTES, GRID_BUILD_WGSL, SNAP_WGSL, physicsWGSL,
  SORT_COUNT_WGSL, SCAN_BLOCKS_WGSL, SCAN_SUMS_WGSL, SORT_SCATTER_WGSL,
} from './shaders.js';

const PARTICLE_BYTES = STRIDE * 4;

async function compile(device, label, code) {
  const module = device.createShaderModule({ label, code });
  const info = await module.getCompilationInfo();
  const errors = info.messages.filter((m) => m.type === 'error');
  if (errors.length) {
    const text = errors.map((m) => `${label}:${m.lineNum}:${m.linePos} ${m.message}`).join('\n');
    throw new Error(`WGSL compile failed\n${text}`);
  }
  return module;
}

export class GpuPhysics {
  /** Compile every pipeline up front so switching broadphase is instant. */
  static async create(device) {
    const engine = new GpuPhysics(device);
    device.pushErrorScope('validation');
    // name: [label, WGSL, entry point]
    const sources = {
      brute: ['physics-brute', physicsWGSL('brute'), 'physicsStep'],
      grid: ['physics-grid', physicsWGSL('grid'), 'physicsStep'],
      sorted: ['physics-sorted', physicsWGSL('sorted'), 'physicsStep'],
      build: ['grid-build', GRID_BUILD_WGSL, 'buildGrid'],
      sortCount: ['sort-count', SORT_COUNT_WGSL, 'countBuckets'],
      scanBlocks: ['sort-scan-blocks', SCAN_BLOCKS_WGSL, 'scanBlocks'],
      scanSums: ['sort-scan-sums', SCAN_SUMS_WGSL, 'scanSums'],
      scatter: ['sort-scatter', SORT_SCATTER_WGSL, 'scatter'],
      snap: ['snap', SNAP_WGSL, 'snapAfterFlip'],
    };
    await Promise.all(Object.entries(sources).map(async ([name, [label, code, entryPoint]]) => {
      const module = await compile(device, label, code);
      engine.pipelines[name] = await device.createComputePipelineAsync({
        label, layout: 'auto', compute: { module, entryPoint },
      });
    }));
    const error = await device.popErrorScope();
    if (error) throw new Error(`WebGPU pipeline error: ${error.message}`);
    return engine;
  }

  constructor(device) {
    this.device = device;
    this.pipelines = {};
    this.count = 0;
    this.broadphase = 'grid';
    this.particleRadius = 0;
    this.gravity = 1;
    this.neckDamping = 0;
    this.containerEulerX = 0;
    this.buffers = null;
    this.current = 0;            // which of the two particle buffers is "now"
    this.readPending = false;
    this.frame = 0;
    this.timing = false;
    this.timingEpoch = 0;
    this.gpuMs = null;
    this.gpuMsMax = 0;
    this.uniformData = new ArrayBuffer(UNIFORM_BYTES);
    this.uniformF32 = new Float32Array(this.uniformData);
    this.uniformU32 = new Uint32Array(this.uniformData);
  }

  get substeps() {
    return gpuSubsteps(this.count);
  }

  /** The buffer holding the latest particle state. */
  get currentBuffer() {
    return this.buffers.particles[this.current];
  }

  /**
   * (Re)allocate for a particle set and upload it.
   * @param {Float32Array} particles STRIDE floats per particle
   */
  configure({ count, particleRadius, profile, broadphase, particles }) {
    this.destroyBuffers();
    const { device } = this;
    this.count = count;
    this.particleRadius = particleRadius;
    this.broadphase = broadphase;
    this.grid = computeGrid(particleRadius, count);
    this.current = 0;
    this.containerEulerX = 0;

    const size = Math.max(count, 1) * PARTICLE_BYTES;
    const particleBuffer = (label) => device.createBuffer({
      label, size,
      // STORAGE: compute shaders read and write it.
      // COPY_SRC: it can be copied to the readback and vertex buffers.
      // COPY_DST: queue.writeBuffer can fill it.
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });

    const buffers = {
      // Two copies, "double buffering". Each substep reads one and writes the
      // other, then they swap. A thread only ever writes its own particle, so
      // no two threads touch the same memory and there is nothing to lock.
      particles: [particleBuffer('particles-A'), particleBuffer('particles-B')],
      uniforms: device.createBuffer({
        label: 'uniforms', size: UNIFORM_BYTES,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      }),
      profile: device.createBuffer({
        label: 'profile-table', size: 256 * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      }),
      // The spatial hash: one list head per bucket, one "next" link per particle.
      bucketHeads: device.createBuffer({
        label: 'bucket-heads', size: this.grid.tableSize * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      }),
      bucketNext: device.createBuffer({
        label: 'bucket-next', size: Math.max(count, 1) * 4,
        usage: GPUBufferUsage.STORAGE,
      }),
      // The sorted grid: a counter per bucket, then each bucket's start
      // (followed by one start per scan block), each grain's rank in its
      // bucket, and the sorted copy with the index each slot came from.
      // COPY_SRC on the last few lets the GPU tests read them.
      bucketCounts: device.createBuffer({
        label: 'bucket-counts', size: this.grid.tableSize * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
      }),
      bucketStarts: device.createBuffer({
        label: 'bucket-starts', size: (this.grid.tableSize + scanBlocks(this.grid.tableSize)) * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
      }),
      rankInBucket: device.createBuffer({
        label: 'rank-in-bucket', size: Math.max(count, 1) * 4,
        usage: GPUBufferUsage.STORAGE,
      }),
      sortedParticles: device.createBuffer({
        label: 'sorted-particles', size,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
      }),
      sortedIds: device.createBuffer({
        label: 'sorted-ids', size: Math.max(count, 1) * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
      }),
      // MAP_READ buffers are the only kind JavaScript can read, and they can't
      // be used by shaders, so reading back always means a GPU-side copy first.
      staging: device.createBuffer({
        label: 'readback-staging', size,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      }),
    };
    this.buffers = buffers;

    device.queue.writeBuffer(buffers.particles[0], 0, particles, 0, count * STRIDE);
    device.queue.writeBuffer(buffers.particles[1], 0, particles, 0, count * STRIDE);
    device.queue.writeBuffer(buffers.profile, 0, profile.buildLookupTable());

    this.#buildBindGroups();
    this.#warmUp();
  }

  // A bind group fills in a pipeline's @binding slots. The read and write
  // buffers swap every substep, so there are two of each: A->B and B->A.
  #buildBindGroups() {
    const { device, buffers, pipelines } = this;
    const entry = (binding, buffer) => ({ binding, resource: { buffer } });
    const [a, b] = buffers.particles;
    const group = (pipeline, entries) =>
      device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries });

    this.bindGroups = {
      brute: [[a, b], [b, a]].map(([read, write]) => group(pipelines.brute, [
        entry(0, buffers.uniforms), entry(1, read), entry(2, write), entry(3, buffers.profile),
      ])),
      grid: [[a, b], [b, a]].map(([read, write]) => group(pipelines.grid, [
        entry(0, buffers.uniforms), entry(1, read), entry(2, write), entry(3, buffers.profile),
        entry(4, buffers.bucketHeads), entry(5, buffers.bucketNext),
      ])),
      build: [a, b].map((read) => group(pipelines.build, [
        entry(0, buffers.uniforms), entry(1, read),
        entry(2, buffers.bucketHeads), entry(3, buffers.bucketNext),
      ])),
      sorted: [[a, b], [b, a]].map(([read, write]) => group(pipelines.sorted, [
        entry(0, buffers.uniforms), entry(1, read), entry(2, write), entry(3, buffers.profile),
        entry(4, buffers.bucketCounts), entry(5, buffers.bucketStarts),
        entry(6, buffers.sortedParticles), entry(7, buffers.sortedIds),
      ])),
      sortCount: [a, b].map((read) => group(pipelines.sortCount, [
        entry(0, buffers.uniforms), entry(1, read),
        entry(2, buffers.bucketCounts), entry(3, buffers.rankInBucket),
      ])),
      scanBlocks: [group(pipelines.scanBlocks, [
        entry(0, buffers.uniforms), entry(1, buffers.bucketCounts), entry(2, buffers.bucketStarts),
      ])],
      scanSums: [group(pipelines.scanSums, [
        entry(0, buffers.uniforms), entry(1, buffers.bucketStarts),
      ])],
      scatter: [a, b].map((read) => group(pipelines.scatter, [
        entry(0, buffers.uniforms), entry(1, read), entry(2, buffers.bucketStarts),
        entry(3, buffers.rankInBucket), entry(4, buffers.sortedParticles), entry(5, buffers.sortedIds),
      ])),
      snap: [a, b].map((particles) => group(pipelines.snap, [
        entry(0, buffers.uniforms), entry(1, particles),
      ])),
    };
  }

  /** Overwrite the particle state from JavaScript (tests use this). */
  upload(particles) {
    for (const buffer of this.buffers.particles) {
      this.device.queue.writeBuffer(buffer, 0, particles, 0, this.count * STRIDE);
    }
  }

  // A GPU driver may not finish compiling a pipeline until it is first used,
  // and that first use then stalls for a noticeable fraction of a second.
  // The flip pipeline is first needed at the instant the flip lands, the
  // worst moment for a hitch, so run it now instead. Twice puts every grain
  // back where it was: (x, -y, -z) and back again.
  #warmUp() {
    if (this.count === 0) return;
    this.#writeUniforms(0);
    const encoder = this.device.createCommandEncoder({ label: 'warm-up' });
    for (let i = 0; i < 2; i++) {
      const pass = encoder.beginComputePass({ label: 'warm-snap' });
      pass.setPipeline(this.pipelines.snap);
      pass.setBindGroup(0, this.bindGroups.snap[this.current]);
      pass.dispatchWorkgroups(Math.ceil(this.count / WORKGROUP_SIZE));
      pass.end();
    }
    this.device.queue.submit([encoder.finish()]);
  }

  setBroadphase(broadphase) {
    this.broadphase = broadphase;
  }

  // Must match `struct Uniforms` in shaders.js, field for field.
  #writeUniforms(subDt) {
    const f = this.uniformF32;
    const u = this.uniformU32;
    f[0] = this.gravity;
    f[1] = PHYSICS.restitution;
    f[2] = PHYSICS.friction;
    f[3] = subDt;
    f[4] = this.containerEulerX;
    f[5] = CHAMBER_BOTTOM;
    f[6] = CHAMBER_TOP;
    f[7] = this.neckDamping;
    f[8] = PHYSICS.neckHalfHeight;
    f[9] = this.grid.cellSize;
    u[10] = this.count;
    u[11] = this.grid.tableMask;
    f[12] = this.grid.origin[0];
    f[13] = this.grid.origin[1];
    f[14] = this.grid.origin[2];
    f[15] = this.frame;   // integer-valued float; scrambles the grid sweep order
    this.device.queue.writeBuffer(this.buffers.uniforms, 0, this.uniformData);
  }

  /**
   * One frame of physics: several substeps, recorded into a single command
   * buffer and submitted once.
   *
   * @param {number} dt
   * @param {GPUBuffer|null} copyTarget if given, the final state is copied
   *   into it on the GPU. This is how "direct" mode feeds the draw call.
   */
  step(dt, copyTarget = null) {
    if (!this.buffers || this.count === 0) return;
    const { device, buffers, pipelines, bindGroups } = this;
    const substeps = this.substeps;
    this.frame = (this.frame + 1) % 65536;
    this.#writeUniforms(dt / substeps);

    const workgroups = Math.ceil(this.count / WORKGROUP_SIZE);
    const bp = this.broadphase;
    const encoder = device.createCommandEncoder({ label: 'physics' });
    // Each dispatch below sees everything the previous one wrote: WebGPU
    // puts a barrier between dispatches that share a buffer, even within
    // one pass.
    const dispatch = (pass, name, groups, x) => {
      pass.setPipeline(pipelines[name]);
      pass.setBindGroup(0, groups);
      pass.dispatchWorkgroups(x);
    };

    for (let s = 0; s < substeps; s++) {
      const now = this.current;
      // Empty the table, then refill it from the current positions.
      if (bp === 'grid') encoder.clearBuffer(buffers.bucketHeads);
      if (bp === 'sorted') encoder.clearBuffer(buffers.bucketCounts);
      const pass = encoder.beginComputePass({ label: `physics-step-${bp}` });
      if (bp === 'grid') {
        dispatch(pass, 'build', bindGroups.build[now], workgroups);
      } else if (bp === 'sorted') {
        dispatch(pass, 'sortCount', bindGroups.sortCount[now], workgroups);
        dispatch(pass, 'scanBlocks', bindGroups.scanBlocks[0], scanBlocks(this.grid.tableSize));
        dispatch(pass, 'scanSums', bindGroups.scanSums[0], 1);
        dispatch(pass, 'scatter', bindGroups.scatter[now], workgroups);
      }
      dispatch(pass, bp, bindGroups[bp][now], workgroups);
      pass.end();
      this.current = 1 - now;
    }

    if (copyTarget) this.#encodeCopy(encoder, copyTarget);
    device.queue.submit([encoder.finish()]);
    this.#sampleGpuTime();
  }

  // A rough clock for the stats readout: how long the GPU takes to finish
  // everything queued up to and including this physics step. When the GPU is
  // keeping up that is close to the cost of the step; when it isn't, it is
  // the backlog. A new measurement starts as soon as the last one lands, so
  // on a GPU that keeps up every frame is timed and a one-frame spike is
  // caught. gpuMsMax is the worst seen since resetTiming().
  #sampleGpuTime() {
    if (this.timing) return;
    this.timing = true;
    const started = performance.now();
    const epoch = this.timingEpoch;
    this.device.queue.onSubmittedWorkDone().then(() => {
      this.timing = false;
      if (epoch !== this.timingEpoch) return;   // reset while in flight
      this.gpuMs = performance.now() - started;
      this.gpuMsMax = Math.max(this.gpuMsMax, this.gpuMs);
    }, () => { this.timing = false; });
  }

  /** Start a fresh "max this run". */
  resetTiming() {
    this.timingEpoch++;
    this.gpuMs = null;
    this.gpuMsMax = 0;
  }

  #encodeCopy(encoder, target) {
    const bytes = Math.min(this.count * PARTICLE_BYTES, target.size);
    encoder.copyBufferToBuffer(this.currentBuffer, 0, target, 0, bytes);
  }

  /** Copy the current state into another GPU buffer without stepping. */
  copyTo(target) {
    if (!this.buffers || this.count === 0) return;
    const encoder = this.device.createCommandEncoder({ label: 'copy-to-vertex-buffer' });
    this.#encodeCopy(encoder, target);
    this.device.queue.submit([encoder.finish()]);
  }

  snapAfterFlip() {
    if (!this.buffers || this.count === 0) return;
    this.containerEulerX = 0;
    this.#writeUniforms(0);
    const encoder = this.device.createCommandEncoder({ label: 'snap' });
    const pass = encoder.beginComputePass({ label: 'snap-after-flip' });
    pass.setPipeline(this.pipelines.snap);
    pass.setBindGroup(0, this.bindGroups.snap[this.current]);
    pass.dispatchWorkgroups(Math.ceil(this.count / WORKGROUP_SIZE));
    pass.end();
    this.device.queue.submit([encoder.finish()]);
  }

  /**
   * Read the particles back into JavaScript.
   *
   * The data arrives at least a frame late: copy to a mappable buffer, ask to
   * map it, and get a promise that resolves once the GPU has caught up. If
   * the previous read hasn't come back yet this one is skipped, so the caller
   * always gets the freshest state that was available, never a backlog.
   *
   * @returns {Promise<boolean>} true if `target` was updated
   */
  async readInto(target) {
    if (!this.buffers || this.count === 0 || this.readPending) return false;
    const { device, buffers } = this;
    const bytes = this.count * PARTICLE_BYTES;
    this.readPending = true;
    try {
      const encoder = device.createCommandEncoder({ label: 'readback' });
      encoder.copyBufferToBuffer(this.currentBuffer, 0, buffers.staging, 0, bytes);
      device.queue.submit([encoder.finish()]);
      await buffers.staging.mapAsync(GPUMapMode.READ, 0, bytes);
      if (buffers !== this.buffers) return false;   // reconfigured while waiting
      target.set(new Float32Array(buffers.staging.getMappedRange(0, bytes)));
      buffers.staging.unmap();
      this.readsCompleted = (this.readsCompleted ?? 0) + 1;
      return true;
    } catch {
      return false;   // buffer destroyed mid-read: harmless
    } finally {
      if (buffers === this.buffers) this.readPending = false;
    }
  }

  /**
   * Bytes of GPU memory allocated, for the stats readout. Every search's
   * buffers exist all the time, so the switch between them is instant.
   */
  get memoryBytes() {
    if (!this.buffers) return 0;
    let total = 0;
    for (const [name, value] of Object.entries(this.buffers)) {
      for (const buffer of name === 'particles' ? value : [value]) total += buffer.size;
    }
    return total;
  }

  destroyBuffers() {
    if (!this.buffers) return;
    const b = this.buffers;
    for (const [name, value] of Object.entries(b)) {
      for (const buffer of name === 'particles' ? value : [value]) buffer.destroy();
    }
    this.buffers = null;
    this.readPending = false;
  }
}
