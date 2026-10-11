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
import { computeGrid } from '../models/grid.js';
import {
  WORKGROUP_SIZE, UNIFORM_BYTES, GRID_BUILD_WGSL, SNAP_WGSL, physicsWGSL,
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
    const [brute, grid, build, snap] = await Promise.all([
      compile(device, 'physics-brute', physicsWGSL('brute')),
      compile(device, 'physics-grid', physicsWGSL('grid')),
      compile(device, 'grid-build', GRID_BUILD_WGSL),
      compile(device, 'snap', SNAP_WGSL),
    ]);
    const pipeline = (label, module, entryPoint) =>
      device.createComputePipelineAsync({ label, layout: 'auto', compute: { module, entryPoint } });
    [engine.pipelines.brute, engine.pipelines.grid, engine.pipelines.build, engine.pipelines.snap] =
      await Promise.all([
        pipeline('physics-brute', brute, 'physicsStep'),
        pipeline('physics-grid', grid, 'physicsStep'),
        pipeline('grid-build', build, 'buildGrid'),
        pipeline('snap', snap, 'snapAfterFlip'),
      ]);
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
    f[15] = 0;
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
    this.#writeUniforms(dt / substeps);

    const workgroups = Math.ceil(this.count / WORKGROUP_SIZE);
    const useGrid = this.broadphase === 'grid';
    const encoder = device.createCommandEncoder({ label: 'physics' });

    for (let s = 0; s < substeps; s++) {
      if (useGrid) {
        // Empty the hash table, then refill it from the current positions.
        encoder.clearBuffer(buffers.bucketHeads);
        const build = encoder.beginComputePass({ label: 'grid-build' });
        build.setPipeline(pipelines.build);
        build.setBindGroup(0, bindGroups.build[this.current]);
        build.dispatchWorkgroups(workgroups);
        build.end();
      }
      const pass = encoder.beginComputePass({ label: 'physics-step' });
      pass.setPipeline(useGrid ? pipelines.grid : pipelines.brute);
      pass.setBindGroup(0, (useGrid ? bindGroups.grid : bindGroups.brute)[this.current]);
      pass.dispatchWorkgroups(workgroups);
      pass.end();
      this.current = 1 - this.current;
    }

    if (copyTarget) this.#encodeCopy(encoder, copyTarget);
    device.queue.submit([encoder.finish()]);
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
      return true;
    } catch {
      return false;   // buffer destroyed mid-read: harmless
    } finally {
      if (buffers === this.buffers) this.readPending = false;
    }
  }

  /** Bytes of GPU memory this configuration uses, for the stats readout. */
  get memoryBytes() {
    if (!this.buffers) return 0;
    const b = this.buffers;
    return b.particles[0].size * 3 + b.bucketHeads.size + b.bucketNext.size + b.profile.size;
  }

  destroyBuffers() {
    if (!this.buffers) return;
    const b = this.buffers;
    for (const buffer of [...b.particles, b.uniforms, b.profile, b.bucketHeads, b.bucketNext, b.staging]) {
      buffer.destroy();
    }
    this.buffers = null;
    this.readPending = false;
  }
}
