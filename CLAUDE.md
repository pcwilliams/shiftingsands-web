# shiftingsands-web: Claude Code Project Reference

The web version of [shiftingsands](https://github.com/pcwilliams/shiftingsands),
an hourglass timer with a real-time granular physics simulation. Sand
grains flow through a procedurally built glass hourglass. Physics runs in
one of three modes (JavaScript, WebGPU compute with readback, WebGPU
compute feeding the draw directly) and everything is drawn by three.js's
WebGPURenderer.

The glass profile, physics kernel, constants, spawn packing and controls
are ported from the Swift app. When the two disagree, the Swift app is the
reference, except for the deliberate differences listed below.

## Owner

- **Developer:** pwilliams (GitHub: pcwilliams)
- **Repo:** `pcwilliams/shiftingsands-web` (public)
- **Hosting:** GitHub Pages via GitHub Actions →
  `https://pcwilliams.github.io/shiftingsands-web/`

## Tech stack

- **No build step.** Plain ES modules plus an import map in `index.html`.
- **three.js r186** (`0.186.1`) from `cdn.jsdelivr.net`, the **WebGPU
  build**: `three` and `three/webgpu` map to `build/three.webgpu.js`, and
  `three/tsl` maps to `build/three.tsl.js`. No addons.
- **Physics:** hand-written **WGSL** compute shaders run through the raw
  WebGPU API (`js/physics/`), not through three.js. three.js and the
  physics share one `GPUDevice`, created in `main.js` and passed to
  `new THREE.WebGPURenderer({ device })`.
- **Persistence:** `localStorage`, keys prefixed `shiftingsands.`.
- **Tests:** Node's built-in runner (`node --test`), 70 tests, zero
  dependencies; plus 18 in-browser GPU tests (`gpu-tests.html`).
- **Zero npm dependencies.** `package.json` exists for `"type": "module"`
  and the `npm test` alias.

## Run locally

```bash
python3 -m http.server 8000
open http://localhost:8000/
node --test
```

## URL parameters (the web's launch arguments)

Applied after `localStorage` loads, so they win. `mode` from the URL is not
persisted; count, size, duration and colours are.

| Param | Example | Effect |
|---|---|---|
| `mode` | `?mode=direct` | `cpu`, `gpu`, `direct`. Aliases: `metal`/`webgpu` → `direct`, `readback` → `gpu` |
| `count` | `?count=50000` | Clamped to the mode's range, rounded to 50 |
| `size` | `?size=1.3` | 0.5 to 1.5 |
| `duration` | `?duration=30` | 5 to 30 |
| `broadphase` | `?broadphase=brute` | `grid` or `brute` (aliases `n2`, `bruteforce`) |
| `colors` | `?colors` | Random colours (`multicolor` also accepted) |
| `stats` | `?stats` | Show the stats readout |
| `autostart` | `?autostart` | Flip after 2 s |
| `test` | `?test` | Duration 10 s, auto-start, summary to `window.shiftingSands.results` and a `[TEST]` console line |
| `seed` | `?seed=4` | Seeds spawn overflow and colours (mulberry32) |

`applyLaunchArgs` sets broadphase, then mode, then count/size, because a
mode change recalls that mode's saved count and size.

## Project structure

```
shiftingsands-web/
├── index.html                 # import map, canvas, four-corner controls
├── gpu-tests.html             # browser page that runs js/dev/gpuSelfTest.js
├── favicon.svg
├── package.json               # "type": "module", "test": "node --test"
├── css/style.css
├── js/
│   ├── main.js                # device request, wiring, frame loop, ?test dump
│   ├── models/                # pure: no DOM, no three.js (all unit-tested)
│   │   ├── hourglassProfile.js  # port of SandGeometry.swift
│   │   ├── packing.js           # radiusForCount, packedParticles, colours, STRIDE
│   │   ├── physicsConfig.js     # constants, MODE_INFO, substeps, slider mapping
│   │   └── grid.js              # spatial hash sizing + CPU reference build
│   ├── physics/
│   │   ├── cpuSimulation.js     # port of GranularSimulation.swift (pure)
│   │   ├── shaders.js           # WGSL: physicsStep (brute/grid), buildGrid, snapAfterFlip
│   │   └── gpuPhysics.js        # port of MetalPhysicsEngine.swift, raw WebGPU
│   ├── scene/hourglassScene.js  # port of HourglassScene.swift, three.js
│   ├── services/
│   │   ├── settingsStore.js     # per-mode count/size, injected Storage
│   │   ├── timerModel.js        # idle → flipping → running → complete
│   │   └── launchArgs.js        # URL parameter parsing
│   ├── ui/controls.js           # DOM ↔ store/timer
│   └── dev/gpuSelfTest.js       # ports of GPUPhysicsTests.swift
├── tests/                     # node:test, *.test.js
├── docs/                      # screenshots for README / HTML docs / og:image
├── .github/workflows/pages.yml
├── README.md
├── architecture.html          # Mermaid diagrams
└── tutorial.html              # how the rendering and GPU physics work
```

## The particle layout (the thing everything shares)

One particle is 8 floats, 32 bytes:

```
[0..2] position   [3] radius   [4..6] velocity   [7] sleep counter
```

`STRIDE = 8` in `packing.js`. The same layout is:

- the `Float32Array` the CPU simulation mutates,
- `struct Particle { posRadius: vec4f, velSleep: vec4f }` in WGSL,
- the instanced vertex buffer (`InstancedInterleavedBuffer`, stride 8)
  that the sand material reads as `instancePosRadius` and
  `instanceVelSleep`.

Because the layouts match, bytes move between the three with no
conversion: `copyBufferToBuffer` from the physics buffer straight into
three.js's vertex buffer.

## Physics modes

| Mode | Label | Range | Default | Where physics runs | How the vertex buffer is filled |
|---|---|---|---|---|---|
| `cpu` | CPU | 50 to 2,000 | 250 | `CpuSimulation.step` | Sim writes the array; `needsUpdate` uploads it |
| `gpu` | GPU | 50 to 20,000 | 5,000 | `GpuPhysics.step` | `readInto(array)` (async, a frame late), then `needsUpdate` uploads it |
| `direct` | Direct | 50 to 200,000 | 10,000 | `GpuPhysics.step` | `copyBufferToBuffer` on the GPU, in the same command buffer as the step |

Brute-force broadphase caps the count at 50,000 (`BRUTE_FORCE_MAX`).
Without WebGPU the renderer is created with `forceWebGL: true` and only
`cpu` is available.

### Substeps (same as the app)

- CPU: 4 up to 500 grains, 3 up to 1,000, 2 above.
- GPU and Direct: 4 up to 1,500, 3 up to 3,000, 2 up to 10,000, 1 above.
  Both broadphases use the same table on purpose, so switching between
  them changes speed and nothing else.

### Flow control (same as the app)

- `gravityScale = max(5 / duration, 0.15)`
- `neckDamping = duration * 0.02`, applied where `|y| < 0.10`
- `dt` is the frame time clamped to 1/30 s. There is no fixed timestep,
  so a 120 Hz display steps twice as finely as a 60 Hz one, as on iOS.

## GPU engine (`gpuPhysics.js` + `shaders.js`)

- **Pipelines** (all compiled once in `GpuPhysics.create`):
  `physics-brute`, `physics-grid`, `grid-build`, `snap`. `layout: 'auto'`.
- **Buffers** per configuration: particles A and B (STORAGE | COPY_SRC |
  COPY_DST), uniforms (64 bytes), profile table (256 floats), bucket
  heads, bucket next-links, readback staging (MAP_READ | COPY_DST).
- **Double buffering:** each substep reads one particle buffer and writes
  the other; `current` flips. Two bind groups per pipeline (A→B, B→A).
- **One submit per frame:** all substeps (and the grid rebuild before
  each) are encoded into one command buffer. Uniforms are written once
  per frame, since `subDt` is the same for every substep.
- **Uniform struct:** 12 scalars then `gridOrigin: vec4f` = 64 bytes.
  `#writeUniforms` must match `struct Uniforms` field for field.
  `particleCount` and `tableMask` are `u32` at float indices 10 and 11.
- **Workgroup size** 64; dispatch `ceil(count / 64)`.
- **Readback** (`readInto`): copy to staging, `mapAsync`, copy out. If a
  read is still pending the call returns `false` instead of queueing.

### Spatial hash broadphase

- Cell size `2r × 1.05` (the 1.05 covers the deep-sleep support test).
- Table size `nextPowerOfTwo(2 × count)`. Each bucket is a **linked
  list**: `bucketHeads` (one `u32` per bucket) and `nextInBucket` (one
  per particle), both storing index + 1 so 0 ends a list. 200,000 grains
  use about 3 MB.
- Hash: `(x·73856093) ^ (y·19349663) ^ (z·83492791) & mask`.
- Build pass: `nextInBucket[tid] = atomicExchange(&bucketHeads[b], tid +
  1)`. Heads are cleared with `encoder.clearBuffer` before each build.
- Lookup: 27 cells; buckets already walked are skipped (two cells can
  share one), then the list is followed to its end. Strangers from other
  cells that share the bucket are rejected by the distance tests.
- **Never give buckets a fixed capacity.** The first version had 8 slots.
  This simulation lets grains overlap heavily under compression: measured
  mid-pour, one cell at the neck held 938 of 10,000 grains. Grains past
  the eighth were dropped, so the grid behaved differently from brute
  force (sand stuck in a pile to one side).
- **Odd grains sweep the 27 cells in reverse** (`sweep = 1 - 2 * (tid &
  1)`). The collision loop updates `pos` as it goes, so visiting order
  biases the result. With one fixed sweep direction every grain got the
  same bias and the settled pile sat 14 radii off-centre at 10,000 grains
  (brute force: 0.07). Alternating brings it under 1 radius. Do not
  remove this, and do not "simplify" the loop order.
- The physics module declares the heads as `array<u32>` (read-only); the
  build module declares the same buffer as `array<atomic<u32>>`
  (read-write). WGSL only allows atomics in read-write storage.
- `js/models/grid.js` is a CPU reference of the same arithmetic. A unit
  test checks it finds every true contact; a GPU test checks that grid
  and brute force produce the same step (largest difference 1e-6 to 3e-6 across runs).

## Scene (`hourglassScene.js`)

- **Sand:** one `Mesh` with an `InstancedBufferGeometry`: an icosphere
  (detail 2 up to 2,000 grains, 1 up to 60,000, 0 above) × `count`
  instances. `MeshLambertNodeMaterial` with
  `positionNode = positionLocal * radius + position`. `frustumCulled =
  false`. Normals are the unit-sphere positions.
- **Glass:** surface of revolution from the outer profile (end discs
  dropped, 96 segments). Unlit `MeshBasicNodeMaterial`, `depthWrite:
  false`, double-sided, opacity from a TSL Fresnel term (0.06 face-on to
  0.26 at grazing angles).
- **Caps:** two cylinders, `MeshStandardNodeMaterial`.
- **Lights:** the app's five-light rig (same directions and colour
  temperatures; intensities set by eye) plus a hemisphere light.
- **Container group** holds glass, caps and sand and rotates for the
  flip; camera and lights are outside it.
- **Camera:** FOV 40°, fitted so the hourglass takes about 62% of the
  height (or half the width on narrow screens).
- **Sleep view:** a TSL uniform mixes a blue tint into grains whose sleep
  counter is above 15 (light) or 30 (deep).

## Timer and control flow

`TimerModel` has no timer of its own; `main.js` calls `tick(now)` every
frame. Two-phase start: `startFlip()` → scene flip (1.2 s, smoothstep) →
`completeFlip(now)`.

- **Start** rebuilds the sand only if mode/count/size/colours changed
  since it was built.
- **Reset** always respawns.
- **Natural completion** leaves the sand where it settled.
- **Mode change** respawns, and restarts if a run was in progress.
- **Broadphase change** is live (no respawn) unless it forced the count
  down, in which case it resets.
- **Count/size/colour change** respawns only when idle; otherwise it
  waits for the next start or reset. Sliders commit on release.

## Deliberate differences from the app

- **Instancing replaces mesh expansion.** The app's `expandMeshes` kernel
  wrote 42 vertices per particle because SceneKit had no route for custom
  instancing. Here the vertex shader does it. There is no equivalent of
  that kernel.
- **Spatial hash broadphase** (new), with brute force kept for comparison.
- **Higher limits:** CPU 2,000 (app 250), Direct 200,000 (app 50,000).
- **Deep-sleep stagger fixed.** The app's counter saturates at 60, after
  which `counter % 30 == tid % 30` is only ever true for threads where
  `tid % 30 == 0`, so most deep sleepers stop checking for support. Here
  the counter cycles 31..60, so every sleeper checks once per 30 steps,
  which is what the app's comments describe.
- **No SSAO, depth of field or shadows.** Flat Lambert lighting only.
- **Logarithmic count slider**, and a live broadphase switch.

## Known behaviour (inherited, not fixed)

- **The sand is not metered to the timer.** Measured in this port: at
  5 s the upper chamber empties in about 1.5 s (CPU and GPU); at 30 s it
  takes 5 s (CPU, 250 grains) to more than 30 s (GPU, 5,000 grains:
71% through after 33 s). Cause:
  see the next item.
- **The neck pinch.** The neck is sized as `r + 0.002` but the wall test
  subtracts a clearance of `0.003`, so the usable radius at the neck is
  negative and every grain there takes the "narrower than the particle"
  branch, which moves it onto the axis. Catmull-Rom overshoot makes the
  pinch tighter still just beside the neck (the inner radius reaches 0 at
  high counts). Grains therefore overlap on the axis instead of queueing,
  and sphere collision pushes some of them sideways out through the
  glass for a frame. A unit test documents the overshoot.
- **Frame-rate-dependent physics** (variable `dt`), as in the app.

## Tests

Node (70): `geometry` (10), `packing` (10), `cpuPhysics` (11, including
ports of the app's CPU tests), `grid` (8), `config` (10), `timer` (8),
`settings` (13).

Browser (18, `gpu-tests.html`; `?only=text` filters by name): gravity, floor, flip transform, drain,
settle-and-sleep, wake-on-support-loss, pile-rests-in-the-middle and a
5,000-grain containment run,
each for `brute` and `grid`, plus "brute force and grid agree" and "a
crowded bucket keeps everyone". The agreement test uses a loose pile and
cannot be made crowded: with many simultaneous contacts the result
depends on visiting order, so the two searches legitimately differ.

Rule of thumb, as in the app: decision logic goes in `js/models` or
`js/services` as pure functions with explicit inputs, and is tested
there. The three.js and WebGPU layers stay thin.

### Headless checks (this is fiddly, read before trying)

Playwright's Chromium can run WebGPU on SwiftShader, with these flags:

```
--enable-unsafe-webgpu --enable-features=Vulkan --use-angle=swiftshader
--use-webgpu-adapter=swiftshader --enable-unsafe-swiftshader
--ignore-gpu-blocklist --use-gl=angle --use-vulkan=swiftshader
--disable-vulkan-surface
```

- Without the last three, compute works but **presenting to a canvas
  loses the device** ("A valid external Instance reference no longer
  exists") on the first or second frame.
- `--enable-unsafe-webgpu` is needed to get an adapter at all, and in
  Chromium 141 it also turns on an older, dictionary-typed
  `GPUTextureViewDescriptor.swizzle`. three r186 passes `swizzle: 'rgba'`
  and every `createView` throws. Work around it in the test harness only,
  with an init script that deletes `swizzle` from the descriptor.
- jsDelivr is blocked in the sandbox: `npm i three@0.186.1` and serve it
  with `context.route('https://cdn.jsdelivr.net/npm/three@0.186.1/**')`.
- SwiftShader runs at 1 or 2 fps. To capture a particular moment, replace
  `shiftingSands.scene.update` with a no-op, call the real one N times
  yourself (awaiting `device.queue.onSubmittedWorkDone()` every few
  steps), wait two animation frames, then screenshot. If the page's own
  loop is left running it double-steps the flip and the frame count
  means nothing.
- **Do not use it to judge frame rates.** No performance number in this
  repo's docs was measured on a real GPU.

`window.shiftingSands = { scene, store, timer, start, reset, adapter,
results }` is exposed for devtools and the harness.

## Deploy

`.github/workflows/pages.yml` runs `node --test` on every push and PR. On
`main` it copies `index.html gpu-tests.html favicon.svg css js docs
architecture.html tutorial.html` into `_site/` and deploys with
`actions/deploy-pages`. **New top-level files must be added to the `cp`
line.**

**One-time repo setup:** Settings → Pages → Build and deployment → Source
→ **GitHub Actions**.

## Open issue: blank start in Direct mode on real browsers

Reported on the first deploy: nothing drawn at start-up in Direct mode
until the first run (one browser) or until switching mode and back
(macOS Safari). Not reproducible in headless Chromium, cause unknown.
Since then: resize is checked every frame, three.js internals are not
touched before the first draw, and `uncapturederror` is shown in the
on-screen notice. If it recurs, the notice text is the first thing to ask
for.

## Gotchas

- **Never give the instance buffer `DynamicDrawUsage`.** three.js
  re-uploads a dynamic attribute from its CPU array on every draw, which
  silently overwrites what the compute pass copied in. Direct mode then
  shows the spawn state forever while the physics runs correctly
  underneath. Default usage uploads only when `needsUpdate` is set.
- **three.js creates the vertex buffer on first draw.** `#vertexBuffer()`
  asks `renderer.backend.get(instances).buffer` only after the sand has
  been drawn once (`sandDrawn`), looks it up fresh every frame, and
  direct mode skips the copy until it exists. This is three.js internals, so recheck
  it when upgrading three.
- **Keep a reference to the `GPUAdapter`** (it is on
  `window.shiftingSands`). Cheap insurance against it being collected.
- **WGSL atomics need read-write storage.** Hence two declarations of the
  bucket-count buffer in two modules.
- **`layout: 'auto'` drops unused bindings.** A bind group must list
  exactly the bindings the entry point statically uses.
- **`queue.writeBuffer` with a typed array takes offset and size in
  elements**, not bytes.
- **`mapAsync` on a buffer that is already pending throws.** `readInto`
  guards with `readPending`.
- **A long-running compute pass can trip the GPU watchdog** (about 2 s on
  Windows) and lose the device. Brute force at 50,000 is 2.5 billion pair
  tests per frame; that is why it is capped there.
- **ES modules need HTTP.** Opening `index.html` from disk fails.
- **Colours are sRGB in the model, linear in the vertex buffer.**
  `linearColors()` converts; skipping it washes the sand out.
- **The glass must not write depth**, or it hides the sand behind it.
