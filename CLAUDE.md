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
- **Tests:** Node's built-in runner (`node --test`), 81 tests, zero
  dependencies; plus 33 in-browser GPU tests (`gpu-tests.html`).
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
| `broadphase` | `?broadphase=sorted` | `grid`, `sorted` or `brute` (aliases `sort`, `counting` → `sorted`; `n2`, `bruteforce` → `brute`) |
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
│   │   ├── grid.js              # spatial hash sizing + CPU reference builds (lists and sorted), sampleSlot
│   │   └── perfHistory.js       # RunTimeline: per-frame timings for one whole run
│   ├── physics/
│   │   ├── cpuSimulation.js     # port of GranularSimulation.swift (pure)
│   │   ├── shaders.js           # WGSL: physicsStep (brute/grid/sorted), buildGrid, sort passes, snapAfterFlip
│   │   └── gpuPhysics.js        # port of MetalPhysicsEngine.swift, raw WebGPU
│   ├── scene/hourglassScene.js  # port of HourglassScene.swift, three.js
│   ├── services/
│   │   ├── settingsStore.js     # per-mode count/size, injected Storage
│   │   ├── timerModel.js        # idle → flipping → running → complete
│   │   └── launchArgs.js        # URL parameter parsing
│   ├── ui/controls.js           # DOM ↔ store/timer
│   ├── ui/perfChart.js          # canvas chart at the bottom of Stats
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
  All three broadphases use the same table on purpose, so switching
  between them changes speed and nothing else.

### Flow control (same as the app)

- `gravityScale = max(5 / duration, 0.15)`
- `neckDamping = duration * 0.02`, applied where `|y| < 0.10`
- `dt` is the frame time clamped to 1/30 s. There is no fixed timestep,
  so a 120 Hz display steps twice as finely as a 60 Hz one, as on iOS.

## GPU engine (`gpuPhysics.js` + `shaders.js`)

- **Pipelines** (all compiled once in `GpuPhysics.create`, from the
  `sources` table there): `physics-brute`, `physics-grid`,
  `physics-sorted`, `grid-build`, `sort-count`, `sort-scan-blocks`,
  `sort-scan-sums`, `sort-scatter`, `snap`. `layout: 'auto'`.
- **Buffers** per configuration: particles A and B (STORAGE | COPY_SRC |
  COPY_DST), uniforms (64 bytes), profile table (256 floats), bucket
  heads, bucket next-links, the sorted grid's five (below), readback
  staging (MAP_READ | COPY_DST). Every search's buffers are allocated
  whatever the broadphase, so the switch is instant; `memoryBytes` is the
  total of all of them.
- **One compute pass per substep.** The clear goes before it (it can't be
  inside a pass), then the build or sort dispatches and the physics
  dispatch share the pass. WebGPU synchronises storage writes between
  dispatches, so each sees the one before.
- **Double buffering:** each substep reads one particle buffer and writes
  the other; `current` flips. Two bind groups per pipeline (A→B, B→A).
- **One submit per frame:** all substeps (and the grid rebuild before
  each) are encoded into one command buffer. Uniforms are written once
  per frame, since `subDt` is the same for every substep.
- **Uniform struct:** 12 scalars then `gridOrigin: vec4f` = 64 bytes
  (`gridOrigin.w` carries the frame number, 0..65535, as a float).
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
- **The walk along one bucket stops after `BUCKET_WALK_LIMIT` (32)
  entries.** The phone's whole-run chart showed physics at 2 ms except
  for about a second starting roughly 1 s after the flip, where steps
  took up to 300 ms: the pour, when thousands of grains sit in a few neck
  cells and every grain nearby walks all of them. With the limit, on the
  software GPU at 10,000 grains, the worst step fell from 580 ms to 370 ms
  (the remainder is the ordinary cost of an awake lower chamber, about
  300 ms there), the drain curve is identical, and the pile settles level
  and fully asleep. A resting pile has at most about ten grains per
  bucket, so the limit does not touch ordinary piles; keep it well above
  that. Nothing is removed from the list: every grain still walks for
  itself, a grain past the limit just isn't seen by neighbours that step.
  This makes grid and brute force differ inside over-full cells.
- **Never give buckets a fixed capacity.** The first version had 8 slots.
  This simulation lets grains overlap heavily under compression: measured
  mid-pour, one cell at the neck held 938 of 10,000 grains. Grains past
  the eighth were dropped, so the grid behaved differently from brute
  force (sand stuck in a pile to one side).
- **The sweep over the 27 cells is scrambled per grain, per frame**
  (`sweepOrder(tid)`: a sign per axis and one of three axis rotations,
  from a hash of `tid` and the frame number in `gridOrigin.w`). The
  collision loop updates `pos` as it goes, so visiting order biases the
  result. Measured on settled piles (surface height spread round the
  pile, in radii / fullest-to-emptiest sector ratio):
  fixed sweep: pile 14 radii off-centre at 10,000;
  odd/even alternation: centred, but 6.9 / 2.62 (a heap on one side);
  scrambled: 0.4 / 1.06; brute force: 0.5 / 1.04.
  Do not remove this, and do not "simplify" the loop order.
- The physics module declares the heads as `array<u32>` (read-only); the
  build module declares the same buffer as `array<atomic<u32>>`
  (read-write). WGSL only allows atomics in read-write storage.
- `js/models/grid.js` is a CPU reference of the same arithmetic. A unit
  test checks it finds every true contact; a GPU test checks that grid
  and brute force produce the same step (largest difference 1e-6 to 3e-6 across runs).

### Sorted grid broadphase (`sorted`, added to compare with `grid` on the phone)

Same cells, hash, table size and walk limit as the grid. Instead of
linked lists, every substep counting-sorts a **copy** of the particles by
bucket, so a bucket is a contiguous range.

- **Passes per substep:** `clearBuffer(bucketCounts)`, then in one pass:
  `countBuckets` (`rank = atomicAdd(&bucketCounts[b], 1)`), `scanBlocks`,
  `scanSums`, `scatter` (`slot = start + rank`, no atomics), physics.
- **Scan:** two levels. `SCAN_BLOCK` = 1,024 buckets per workgroup of 256
  threads, each loading four counters as a `vec4u` (the table size is a
  power of two, so it divides by four); Hillis-Steele in workgroup memory.
  `bucketStarts` is `tableSize` starts-within-block followed by one entry
  per block; `scanBlocks` writes the block totals there and `scanSums`
  (one workgroup, a serial run per thread, up to 512 blocks at 200,000)
  turns them into block starts. A bucket's start is
  `bucketStarts[b] + bucketStarts[tableSize + b / SCAN_BLOCK]`. The first
  version used 256 buckets per block; on SwiftShader the block scan cost
  40 ms a step at 10,000 grains (barrier emulation), 10 ms with 1,024.
- **Identity is kept.** The particle buffers are never reordered:
  `sortedParticles` is a snapshot copy and `sortedIds[slot]` is the grain
  it came from. Results go to `particlesOut[tid]`. Colours, the `tid % 30`
  deep-sleep stagger, `sweepOrder(tid)` and the tests all keep working.
- **Threads run in sorted order:** `let tid = sortedIds[gid.x]` at the top
  of the kernel, so a workgroup handles grains from a few cells. The rest
  of the kernel is unchanged.
- **Reading a bucket:** `min(n, BUCKET_WALK_LIMIT)` visits from a random
  start (`sampleOffset(tid, bucket)`, scaled by `n` in float to avoid an
  integer division). If `n` fits, every slot in rotated order; if not,
  visits spread evenly along the range (`sampleSlot` in `grid.js` is the
  same sum, and is unit-tested). Only the crowded case pays the divisions.
- **Bindings:** the sorted physics kernel uses 7 storage buffers (the
  default per-stage limit is 8). That is why block starts share the
  `bucketStarts` buffer instead of having their own.
- **Measured on SwiftShader at 10,000 grains** (meaningless for speed on
  real hardware, but the behaviour is identical): same drain curve, same
  settled pile (level 0.4 radii), all asleep. Physics dispatch alone
  roughly even with the list grid mid-pour; the sort passes add a
  baseline that is mostly barrier emulation. No phone numbers yet.
- When benchmarking stages in isolation, run clear + build + physics
  together each time. Repeating `countBuckets` without the clear inflates
  every count and makes the sorted physics look 2x slower than it is
  (this happened once).

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

## Stats panel and timing chart

`#stats-panel` holds the text readout, a canvas chart and its legend.

- **Physics time (GPU modes):** `GpuPhysics.#sampleGpuTime` times from the
  physics submit to `queue.onSubmittedWorkDone()`. A new sample starts as
  soon as the last lands, so every frame is timed when the GPU keeps up.
  `gpuMsMax` is the worst since `resetTiming()` (called on start and on
  rebuild). It is queue-drain time, so under backlog it reads the backlog.
- **Physics time (CPU mode):** wall time of `sim.step`.
- **Drawing time:** `main.js` times the whole frame's GPU work the same
  way and subtracts physics. Not available on the WebGL fallback.
- **The chart shows one whole run and does not scroll.** `RunTimeline`
  (`models/perfHistory.js`) begins at Start with a fixed span of flip +
  duration + `SETTLE_TAIL` (3 s) and records until the span is used up.
  `PerfChart` fills it in from the left: 1 px lines, one point per pixel
  column holding the worst frame in that column (so a single slow frame
  survives), ticks where the flip lands and where the countdown reaches
  0:00, a dashed line at 16.7 ms. The value axis is a square root
  (`sqrtPosition`) so a few-millisecond baseline and a 260 ms spike are
  both readable; its top is `niceCeiling` of the run's maximum. A rebuild
  clears it. Colours are the dataviz skill's validated dark-surface slots
  1 and 2 (`#3987e5` drawing, `#d95926` physics); values are in the
  legend as text.
- The Stats, Sleep view and Colours boxes have `autocomplete="off"`, and
  `Controls` applies the boxes' actual state at start-up, because browsers
  can restore a ticked box on reload without firing `change`.

- `BUILD` in `main.js` is printed on the first Stats line. It is `'dev'`
  in the repo; the deploy workflow replaces it with the Singapore time and
  the short commit (for example `12 Oct 09:40 1a2b3c4`).

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
- **Spatial hash broadphase** (new), in two forms (linked lists, and
  counting-sorted), with brute force kept for comparison.
- **Higher limits:** CPU 2,000 (app 250), Direct 200,000 (app 50,000).
- **Deep-sleep stagger fixed.** The app's counter saturates at 60, after
  which `counter % 30 == tid % 30` is only ever true for threads where
  `tid % 30 == 0`, so most deep sleepers stop checking for support. Here
  the counter cycles 31..60, so every sleeper checks once per 30 steps,
  which is what the app's comments describe.
- **Support means support from underneath.** In the app, a grain counts
  as supported if it touches any grain (`contactCount > 0`, and "any
  neighbour within 1.05x" in the deep-sleep check). Grains fall asleep in
  the pouring column, lose what was under them, and the leftover clump
  holds itself up for ever (seen on an iPhone at 20,000 grains; a
  constructed sleeping clump hangs in both searches). Here
  `hasSupport` and the deep-sleep check require the floor (`onFloor`),
  glass sloping steeply inwards below the grain (`onGlassSlope`: more
  than half a radius per radius of height, which only the rim by the
  floor is; a gentler threshold stuck grains to the side of the bulb
  below its widest point), or a grain
  whose centre is lower by more than a quarter of the separation
  (`holdsMeUp`). Piles still settle fully (19,999 of 20,000 asleep).
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

Node (81): `perfHistory` (8), `geometry` (10), `packing` (10), `cpuPhysics` (11, including
ports of the app's CPU tests), `grid` (11: includes the sorted layout,
sorted vs list candidates, and sampleSlot spread), `config` (10), `timer` (8),
`settings` (13).

Browser (33, `gpu-tests.html`; `?only=text` filters by name): gravity, floor, flip transform, drain,
settle-and-sleep, wake-on-support-loss, mid-air-clump-falls,
nothing-sticks-to-the-side-wall,
pile-rests-level-and-centred and a
5,000-grain containment run,
each for `brute`, `grid` and `sorted`, plus "all three searches agree",
"a crowded bucket keeps everyone" (both grids; 31 grains in one bucket,
under the walk limit) and "the sort puts every grain in its bucket once"
(140,000 grains, 512 scan blocks; reads the sort buffers back, which is
why they have COPY_SRC). The agreement test uses a loose pile and
cannot be made crowded: with many simultaneous contacts the result
depends on visiting order, so the searches legitimately differ.

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
- **Do not use it to judge frame rates.** Real-hardware numbers come only
  from the user's phone (the iPhone section below). SwiftShader emulates
  `workgroupBarrier` on the CPU, so anything barrier-heavy (the sorted
  grid's scan) looks far worse there than on a GPU.
- `?only=sort` on `gpu-tests.html` runs just the sorted-grid tests.

`window.shiftingSands = { scene, store, timer, start, reset, adapter,
results }` is exposed for devtools and the harness.

## Deploy

`.github/workflows/pages.yml` runs `node --test` on every push and PR. On
`main` it assembles `_site/` and deploys with `actions/deploy-pages`.

- `index.html gpu-tests.html favicon.svg docs architecture.html
  tutorial.html` are copied to the root. **New top-level files must be
  added to that `cp` line.**
- **`css/` and `js/` are copied into `v/<short sha>/`**, and `sed`
  rewrites the stylesheet and script references in `index.html` and the
  import in `gpu-tests.html` to point there. Pages serves everything
  with a ten-minute cache and offers no way to change that; the user
  repeatedly ran a stale or mixed copy on iOS Safari. With a new folder
  per commit every script and stylesheet has a new URL each deploy, and
  relative imports keep the whole module graph inside one folder.
- The same step stamps `BUILD`. Four `grep -q` lines fail the deploy if
  a rewrite stops matching, so **if you change those three references or
  the `BUILD` line, change the `sed` patterns with them.**
- Locally nothing changes: the repo's own `index.html` still points at
  `js/` and `css/`.

**One-time repo setup:** Settings → Pages → Build and deployment → Source
→ **GitHub Actions**.

## iPhone 16 Pro: 300 ms physics steps during the pour (fixed, confirmed at 20,000)

At 20,000 grains (Direct, grid) the phone ran physics in about 2 ms and
drawing in about 7 ms, except for about a second of each run where steps
took up to 300 ms. The whole-run chart placed it roughly 1 s after the
flip lands and lasting about 1 s, so it is the pour, not a one-off
compile stall (`#warmUp` for the snap pipeline was added on that theory
and is harmless). `BUCKET_WALK_LIMIT` is the response; see the spatial
hash section. Confirmed on the phone at 20,000 grains (build 6b0b7e9):
physics peaks at 15 ms, flat through the pour, pile settles clean.

At 100,000 grains on the same phone (same build) it struggles in two
places: about 40 to 60 ms per step during the flip (every grain awake and
tumbling) and a peak of 169 ms in the pour, about 3 ms once settled.
Response so far: the `sorted` broadphase, added as a third option so the
whole-run chart can compare it with `grid` on the phone before anything
is replaced. Awaiting those numbers. Other candidates: skipping the
light-sleep neighbour scan where possible; fewer triangles per grain;
reusing the previous substep's sort order.

The "drawing" spikes in the
same window are an artefact: drawing is whole-frame drain time minus a
physics sample from a different frame, so it swings between 0 and about
100 ms while the queue is backed up.

Things tried that did not help: a larger hash table (visits 40 to 35);
capping contacts per grain at 48 (the cost is grains near a crowded cell
walking it, not grains in it making contacts).

## Releases

- **v1.0.0** = `6b0b7e9` (grid with walk limit, support-from-below,
  per-commit deploy folder). The session couldn't push tags (403), so the
  user creates it on GitHub.

## Open issue: blank start in the GPU modes on macOS Safari

Reported repeatedly: on macOS Safari nothing is drawn at start-up in GPU
or Direct mode until the user switches to CPU mode and back. Not
reproducible in headless Chromium; cause unknown. What CPU mode has that
the others lack: no compute submissions, and a `writeBuffer` upload of
the instance buffer every frame.

Mitigations so far, none confirmed: resize checked every frame; three.js
internals untouched before the first draw; `uncapturederror` shown in the
notice; `WARMUP_FRAMES` (4 draws after each rebuild before any compute is
queued). Stats now ends with a plumbing line (backend, frames drawn,
whether the GPU copy is bound or how many readbacks completed) and any
WebGPU error. If it recurs, ask for: that Stats line, whether the glass is
visible, and the Web Inspector console.

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
