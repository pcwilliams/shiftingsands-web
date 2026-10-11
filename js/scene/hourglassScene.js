// The three.js scene: a port of HourglassScene.swift.
//
// three.js draws everything here through its WebGPURenderer. The interesting
// part is the sand. All three physics modes feed the same draw call:
//
//   one small sphere mesh  x  N instances
//
// "Instancing" means the GPU draws the same mesh N times and hands the vertex
// shader a different row of a per-instance buffer each time. That buffer is
// the particle array itself (8 floats each), so the vertex shader only has to
// do   worldPosition = spherePoint * radius + particlePosition.
//
// The iOS app had to expand every particle into 42 vertices in a compute
// kernel because SceneKit had no way in for custom instancing. Here the
// vertex shader does that expansion for free.
//
// How the instance buffer gets its data is the whole difference between the
// modes:
//   cpu     JavaScript writes the array; three.js uploads it each frame.
//   gpu     Compute writes a GPU buffer; it is read back into the array
//           (asynchronously, a frame late); three.js uploads it again.
//   direct  Compute writes a GPU buffer; one GPU-side copy puts it in the
//           vertex buffer. JavaScript never sees a particle.

import * as THREE from 'three';
import { attribute, float, mix, normalView, positionLocal, positionViewDirection, step, uniform, vec3 } from 'three/tsl';

import {
  createProfile, createRevolutionSurface, neckRadiusForParticle,
} from '../models/hourglassProfile.js';
import { STRIDE, radiusForCount, packedParticles, particleColors, mulberry32 } from '../models/packing.js';
import { PHYSICS, sphereDetail, easeInOut } from '../models/physicsConfig.js';
import { CpuSimulation } from '../physics/cpuSimulation.js';
import { GpuPhysics } from '../physics/gpuPhysics.js';

const FOV = 40;
const WARMUP_FRAMES = 4;
const LOOK_AT = new THREE.Vector3(0, -0.02, 0);
const HOURGLASS_HEIGHT = 1.05;
const HOURGLASS_WIDTH = 0.48;

export class HourglassScene {
  /**
   * @param {HTMLCanvasElement} canvas
   * @param {GPUDevice|null} device  a device we created, or null to fall back
   *   to WebGL 2 (CPU mode only)
   */
  static async create(canvas, device) {
    const scene = new HourglassScene(canvas, device);
    await scene.renderer.init();
    scene.fittedWidth = -1;   // size again now the backend exists
    if (device) scene.engine = await GpuPhysics.create(device);
    return scene;
  }

  constructor(canvas, device) {
    this.canvas = canvas;
    this.device = device;
    this.engine = null;
    this.sim = null;
    this.mode = 'cpu';
    this.count = 0;
    this.particleRadius = 0;
    this.flip = null;
    this.gpuVertexBuffer = null;

    // Passing our own device means three.js and the physics engine share one
    // GPU context, which is what lets a compute result be drawn with no copy
    // through JavaScript.
    this.renderer = new THREE.WebGPURenderer(
      device ? { canvas, antialias: true, device } : { canvas, antialias: true, forceWebGL: true },
    );
    this.renderer.setPixelRatio(Math.min(globalThis.devicePixelRatio || 1, 2));

    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x070605);
    this.camera = new THREE.PerspectiveCamera(FOV, 1, 0.1, 20);

    // Everything that turns over during the flip hangs off this group.
    // Camera and lights stay outside it, so they don't rotate.
    this.container = new THREE.Group();
    this.scene.add(this.container);

    this.sleepView = uniform(0);
    this.#buildLights();
    this.#buildCaps();
    this.glassMaterial = this.#createGlassMaterial();
    this.sandMaterial = this.#createSandMaterial();
    this.glass = null;
    this.sand = null;
    this.resize();
  }

  // ── Static parts ───────────────────────────────────────────────────────

  // The app's five-light rig, with the same directions and colour
  // temperatures. SceneKit's intensities don't map to three.js units, so the
  // ratios are kept and the overall level was set by eye.
  #buildLights() {
    const directional = (color, intensity, x, y, z) => {
      const light = new THREE.DirectionalLight(color, intensity);
      light.position.set(x, y, z);
      this.scene.add(light);
    };
    directional(0xffeedd, 2.20, 1.0, 2.0, 1.5);    // key: warm, upper right
    directional(0xffe9d2, 1.85, 0.0, 0.3, 2.0);    // front fill, from the camera
    directional(0xf4f6ff, 0.75, -1.5, 0.5, 1.0);   // cool fill, from the left
    directional(0xffeccf, 0.75, 0.5, 1.0, -1.5);   // rim, from behind
    const bottom = new THREE.PointLight(0xffdbba, 0.12, 3, 1);
    bottom.position.set(0, -0.8, 0.5);
    this.scene.add(bottom);
    this.scene.add(new THREE.HemisphereLight(0x3a3835, 0x0d0c0b, 1.0));
  }

  #buildCaps() {
    const material = new THREE.MeshStandardNodeMaterial({
      color: new THREE.Color().setRGB(0.45, 0.30, 0.18, THREE.SRGBColorSpace),
      metalness: 0.1,
      roughness: 0.7,
    });
    const geometry = new THREE.CylinderGeometry(0.18, 0.18, 0.025, 64);
    for (const y of [0.5125, -0.5125]) {
      const cap = new THREE.Mesh(geometry, material);
      cap.position.y = y;
      this.container.add(cap);
    }
  }

  // Glass is mostly invisible face-on and brighter at glancing angles (the
  // Fresnel effect). TSL, three.js's shading language, lets that be written
  // as an expression: it compiles to WGSL on WebGPU and GLSL on WebGL.
  #createGlassMaterial() {
    // Unlit on purpose. The app found that lit, reflective glass always came
    // out too bright; a faint even veil with brighter edges reads as glass.
    const material = new THREE.MeshBasicNodeMaterial({
      color: 0xfff4e6,
      transparent: true,
      side: THREE.DoubleSide,
      depthWrite: false,   // otherwise the glass would hide the sand behind it
    });
    const facing = normalView.dot(positionViewDirection).abs();
    const fresnel = float(1).sub(facing).pow(3);
    material.opacityNode = mix(float(0.06), float(0.26), fresnel);
    return material;
  }

  #createSandMaterial() {
    // Lambert: diffuse only. A specular term blows the sand out to white.
    const material = new THREE.MeshLambertNodeMaterial();
    const posRadius = attribute('instancePosRadius', 'vec4');
    const velSleep = attribute('instanceVelSleep', 'vec4');
    const tint = attribute('instanceColor', 'vec3');

    // The whole of "mesh expansion": scale a unit sphere, move it into place.
    material.positionNode = positionLocal.mul(posRadius.w).add(posRadius.xyz);

    // Optional debug view: light sleepers pale blue, deep sleepers dark blue.
    const asleep = step(15.5, velSleep.w);
    const deep = step(30.5, velSleep.w);
    const sleepColor = mix(vec3(0.35, 0.62, 0.95), vec3(0.06, 0.16, 0.5), deep);
    material.colorNode = mix(tint, sleepColor, asleep.mul(this.sleepView));
    return material;
  }

  #rebuildGlass(profile) {
    if (this.glass) {
      this.container.remove(this.glass);
      this.glass.geometry.dispose();
    }
    // Drop the flat end discs: the wooden caps cover them.
    const surface = createRevolutionSurface(profile.outer.slice(1, -1), 96);
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(surface.positions, 3));
    geometry.setAttribute('normal', new THREE.BufferAttribute(surface.normals, 3));
    geometry.setAttribute('uv', new THREE.BufferAttribute(surface.uvs, 2));
    geometry.setIndex(new THREE.BufferAttribute(surface.indices, 1));
    this.glass = new THREE.Mesh(geometry, this.glassMaterial);
    this.glass.renderOrder = 1;
    this.container.add(this.glass);
  }

  // ── Particles ──────────────────────────────────────────────────────────

  /**
   * Build (or rebuild) the sand for a mode. Resizes the glass neck to suit.
   * Falls back to CPU mode if WebGPU compute isn't available.
   */
  setParticles({ mode, count, size, randomColors, broadphase, seed = null }) {
    this.cancelFlip();
    if (mode !== 'cpu' && !this.engine) mode = 'cpu';
    this.mode = mode;
    this.count = count;
    this.broadphase = broadphase;

    const random = seed == null ? Math.random : mulberry32(seed);
    const radius = radiusForCount(count, size);
    this.particleRadius = radius;

    // The neck is rebuilt so that one particle just fits through.
    const profile = createProfile(neckRadiusForParticle(radius));
    this.profile = profile;
    this.#rebuildGlass(profile);

    if (this.sand) {
      this.container.remove(this.sand);
      this.sand.geometry.dispose();
    }

    // Per-instance data: the particle array, viewed as two vec4 attributes.
    const data = new Float32Array(count * STRIDE);
    const instances = new THREE.InstancedInterleavedBuffer(data, STRIDE);
    // Deliberately not DynamicDrawUsage: three.js re-uploads a dynamic buffer
    // from this array on every draw, which would overwrite what the compute
    // pass copied in. With the default usage it uploads only when
    // needsUpdate is set, which CPU and GPU modes do and direct mode doesn't.
    this.instances = instances;
    this.gpuVertexBuffer = null;
    this.sandDrawn = false;
    this.framesSinceBuild = 0;

    const sphere = new THREE.IcosahedronGeometry(1, sphereDetail(count));
    const geometry = new THREE.InstancedBufferGeometry();
    geometry.setAttribute('position', sphere.getAttribute('position'));
    // On a unit sphere the normal at a point is the point itself.
    geometry.setAttribute('normal', sphere.getAttribute('position'));
    geometry.setAttribute('instancePosRadius', new THREE.InterleavedBufferAttribute(instances, 4, 0));
    geometry.setAttribute('instanceVelSleep', new THREE.InterleavedBufferAttribute(instances, 4, 4));
    geometry.setAttribute('instanceColor', new THREE.InstancedBufferAttribute(linearColors(count, randomColors, random), 3));
    geometry.instanceCount = count;
    this.triangles = (sphere.getAttribute('position').count / 3) * count;

    this.sand = new THREE.Mesh(geometry, this.sandMaterial);
    // Every instance has moved in the vertex shader, so the mesh's own
    // bounding sphere says nothing useful. Never cull it.
    this.sand.frustumCulled = false;
    this.container.add(this.sand);

    if (mode === 'cpu') {
      this.engine?.destroyBuffers();
      this.sim = new CpuSimulation({ count, particleRadius: radius, profile, target: data, random });
    } else {
      this.sim = null;
      data.set(packedParticles(count, radius, profile, random));
      this.engine.configure({ count, particleRadius: radius, profile, broadphase, particles: data });
    }
    instances.needsUpdate = true;
    return mode;
  }

  setBroadphase(broadphase) {
    this.broadphase = broadphase;
    this.engine?.setBroadphase(broadphase);
  }

  setSleepView(on) {
    this.sleepView.value = on ? 1 : 0;
  }

  get substeps() {
    return this.mode === 'cpu' ? this.sim.substeps : this.engine.substeps;
  }

  // three.js creates the GPU vertex buffer the first time the mesh is drawn.
  // After that we can ask the backend for it and copy into it directly.
  // Nothing is asked of three.js's internals until it has drawn this sand at
  // least once, and the answer is looked up fresh each frame rather than
  // kept, so a buffer three.js replaced can never be written to by mistake.
  #vertexBuffer() {
    if (!this.sandDrawn) return null;
    this.gpuVertexBuffer = this.renderer.backend.get(this.instances)?.buffer ?? null;
    return this.gpuVertexBuffer;
  }

  // ── Flip ───────────────────────────────────────────────────────────────

  /** Turn the hourglass over. `onDone` fires when the sand should start timing. */
  startFlip(onDone) {
    this.flip = { elapsed: 0, onDone };
  }

  cancelFlip() {
    this.flip = null;
    this.container.rotation.x = 0;
    if (this.sim) this.sim.containerEulerX = 0;
    if (this.engine) this.engine.containerEulerX = 0;
  }

  #advanceFlip(dt) {
    const flip = this.flip;
    if (!flip) return;
    flip.elapsed += dt;
    if (flip.elapsed < PHYSICS.flipDuration) {
      this.container.rotation.x = Math.PI * easeInOut(flip.elapsed / PHYSICS.flipDuration);
      return;
    }
    // Landed. The glass looks the same upside down, so snap the container
    // back to zero and rotate the particles instead: nothing visibly changes.
    this.flip = null;
    this.container.rotation.x = 0;
    if (this.mode === 'cpu') {
      this.sim.snapAfterFlip();
      this.instances.needsUpdate = true;
    } else {
      this.engine.snapAfterFlip();
      if (this.mode === 'direct') {
        const target = this.#vertexBuffer();
        if (target) this.engine.copyTo(target);
      }
    }
    flip.onDone?.();
  }

  // ── Per frame ──────────────────────────────────────────────────────────

  /**
   * Advance the flip and the physics by one frame.
   * @param {number} dt seconds, already clamped
   */
  update(dt, { gravity, neckDamping }) {
    // Let three.js draw new sand a few times before any compute work is
    // queued. On macOS Safari nothing was drawn at start-up in the GPU modes
    // until the page had spent a while in CPU mode, where no compute runs.
    // The cause isn't known; this gives every start the same quiet first
    // frames that made it work. The sand is at rest then, so nothing is lost.
    if (this.framesSinceBuild < WARMUP_FRAMES) return;
    this.#advanceFlip(dt);
    const angle = this.container.rotation.x;

    if (this.mode === 'cpu') {
      const sim = this.sim;
      sim.gravity = gravity;
      sim.neckDamping = neckDamping;
      sim.containerEulerX = angle;
      sim.step(dt);
      this.instances.needsUpdate = true;          // upload the array
      return;
    }

    const engine = this.engine;
    engine.gravity = gravity;
    engine.neckDamping = neckDamping;
    engine.containerEulerX = angle;

    if (this.mode === 'direct') {
      engine.step(dt, this.#vertexBuffer());      // compute, then GPU-side copy
      return;
    }

    // 'gpu': compute, read back, upload again.
    engine.step(dt);
    const instances = this.instances;
    engine.readInto(instances.array).then((updated) => {
      if (updated && instances === this.instances) instances.needsUpdate = true;
    });
  }

  render() {
    this.resizeIfNeeded();
    this.renderer.render(this.scene, this.camera);
    this.sandDrawn = true;
    this.framesRendered = (this.framesRendered ?? 0) + 1;
    this.framesSinceBuild++;
  }

  /**
   * Resize if the canvas's on-screen size or the pixel ratio has changed.
   * Called every frame: it is three comparisons, and it catches everything a
   * 'resize' event can miss (a canvas with no layout yet at start-up, mobile
   * browser bars sliding in and out, dragging between displays).
   */
  resizeIfNeeded() {
    const ratio = Math.min(globalThis.devicePixelRatio || 1, 2);
    if (this.canvas.clientWidth !== this.fittedWidth ||
        this.canvas.clientHeight !== this.fittedHeight || ratio !== this.fittedRatio) {
      this.resize();
    }
  }

  /** Fit the hourglass to the canvas, leaving room for the controls. */
  resize() {
    this.fittedWidth = this.canvas.clientWidth;
    this.fittedHeight = this.canvas.clientHeight;
    this.fittedRatio = Math.min(globalThis.devicePixelRatio || 1, 2);
    if (!this.fittedWidth || !this.fittedHeight) return;   // not laid out yet; try next frame
    const width = this.fittedWidth;
    const height = this.fittedHeight;
    this.renderer.setPixelRatio(this.fittedRatio);
    this.renderer.setSize(width, height, false);
    const aspect = width / height;
    this.camera.aspect = aspect;

    const halfTan = Math.tan(THREE.MathUtils.degToRad(FOV / 2));
    const fitHeight = (HOURGLASS_HEIGHT / 0.62) / (2 * halfTan);
    const fitWidth = (HOURGLASS_WIDTH / 0.5) / (2 * halfTan * aspect);
    const distance = Math.max(fitHeight, fitWidth);
    this.camera.position.set(0, 0.05 * (distance / 3), distance);
    this.camera.lookAt(LOOK_AT);
    this.camera.updateProjectionMatrix();
  }

  /** Current particle data, for ?test and the stats readout. */
  async readParticles() {
    if (this.mode === 'cpu') return this.sim.data.slice();
    const out = new Float32Array(this.count * STRIDE);
    for (let attempt = 0; attempt < 20; attempt++) {
      if (await this.engine.readInto(out)) return out;
      await new Promise((resolve) => setTimeout(resolve, 4));
    }
    return out;
  }
}

/** sRGB bytes to the linear floats the renderer lights with. */
function linearColors(count, randomColors, random) {
  const bytes = particleColors(count, randomColors, random);
  const out = new Float32Array(count * 3);
  const color = new THREE.Color();
  for (let i = 0; i < count; i++) {
    color.setRGB(bytes[i * 4] / 255, bytes[i * 4 + 1] / 255, bytes[i * 4 + 2] / 255, THREE.SRGBColorSpace);
    out[i * 3] = color.r;
    out[i * 3 + 1] = color.g;
    out[i * 3 + 2] = color.b;
  }
  return out;
}
