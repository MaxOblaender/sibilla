# Plan — Speeding up the Zernike fit (`fit_3d`)

Status: **O1 is the DEFAULT** (`FitInput3D.bank_policy = "cluster"` since
2026-11-17, commit after a179b62). Gate passed: 3 real clusters (T/D_clean/
E_clean, max |ΔZ| ≤ 1.9e-9, identical decisions, full_box arms reproduce the
recorded campaign coefficients) + 11 synthetic fit-level A/Bs (7 real
sub-box runs, 3 fallbacks, 1 corner-clamped sub-box; max |ΔZ| ≤ 4.6e-10).
Legacy path stays available via `bank_policy="full_box"` (bit-identical).
O2–O5 not implemented. Measured numbers: 2026-10-30 (real channel timings),
2026-11-16/17 (probe, nodeasb01g benchmark, gate, synth batch).

## 0. Where the time goes (measured)

Per `fit_3d` outer evaluation at box (nz, M, N), `nseg` rods, `nzcand` z-candidates:

| term | cost | share |
|---|---|---|
| `cyclic3d` per rod × per z-candidate — 3D FFT over the **whole box** | `nseg × nzcand` FFTs, ∝ box voxels | **~90%** |
| NNLS (`nz·L` rows × `nseg+nz` cols) | ∝ box volume (rows) | ~7% |
| PSF volume rebuild (`nz` 2D FFTs of G²) | ∝ nz·G² | ~2% |
| × outer least_squares (evals × starts × sweeps) | multiplier | |

Measured per-eval totals (confocal, 7 z-cands):

| box | per eval | ×~150 evals |
|---|---|---|
| crop1 (26×196×196) | 2.1 s | ~5 min |
| local cluster (25×160×256) | 0.47 s | ~1.2 min |
| region boxes, 551²–893² pad | ~4 s | 30–60 min |
| whole-field (142×1547²) | ~416 s | ~17 h |

Two facts that shape all options below:
1. **One fit is single-threaded** (numpy FFT + scipy NNLS are serial).
2. **Cost scales with box volume, not rod count.** Adding rods is cheap;
   enlarging the box is not.

### 0.1 Week-end runs: what the input boxes actually were

Source: nodeasb01 `/asbdata/moblaende26/zerniky/weekend_logs/{fit_*.json,*.log}`,
driver `server_fit.py` (`n_multi_start=2`, `max_nfev=150`, confocal model,
7 z-candidates) — i.e. **3 starts × 3 outer sweeps**, and the per-rod z search.

| run | padded box (nz×M×N) | rods | wall | s per Mvox |
|---|---|---|---|---|
| tright | 30×245×245 | 3 | 24.8 min | 826 |
| region4 | 23×158×158 | 2 | 47.9 min | 5005 |
| region6 | 35×536×536 | 2 | 108.9 min | 650 |
| merged (old box) | 43×460×460 | 5 | 188.1 min | 1240 |
| merged (99%-margin) | 43×551×551 | 5 | 288.0 min | 1324 |
| region5 (99%-margin) | 52×893×893 | **2** | **364.3 min** | 527 |
| region3 (old box) | 36×320×320 | 7 | 376.1 min | 6121 |
| region3 (99%-margin) | 36×393×393 | 7 | 288.0 min | 3108 |

Live status (2026-11-02): the region3 rerun was still running as of
~14:00 server time (box 36×393×393, 7 rods, 99 %-margin) — no result yet.
The `allrods` chained fit (22 rods, one box) was cancelled before it
started: a single whole-field fit is both ~80 h and the wrong question if
the field is aberration-varying (see §2.5 and `FIELDTEST_one_set_vs_Zmap.md`).
Also measured 2026-11-02: the region3 process used ≈3.1 CPU cores in
6.65 h wall (CPU time 20.7 h) — the NNLS/BLAS part IS threaded under
OpenMP, so "one fit is single-threaded" below applies to the FFT/render
loop, not to the linear algebra.

Read-out (these overturn the "cost = box volume" assumption at small size):

- **Time is NOT a clean function of volume alone** (s/Mvox spans 11×). The
  cheap-per-voxel runs are the large boxes; the small-box fits (region4,
  region3) cost far more *per voxel* — consistent with a near-constant,
  eval-dominated overhead per fit (NNLS + PSF rebuild + renders) that only
  amortises in big boxes.
- **The 99 %-energy guard band is applied in X AND Y to the rod *bounding
  rectangle*.** region5: 2 rods ~800 px apart → 893×893×52 box for two rods.
  That is the single largest avoidable cost in the week-end set.
- Rod count is **not** the driver: region5 (2 rods) took 364 min; region6
  (2 rods) 109 min — the difference is box size, not rods.
- Whole channel (196×1440×1920 = 542 Mvox, 4.3 GB float64): by extrapolation
  of the large-box rate (~530 s/Mvox) one fit is **≈ 80 h** — and memory per
  FFT component would be ~8.7 GB complex. Whole-field fitting is therefore a
  *model* change (per-rod support), not a tuning change.

## 1. Options (ranked by win ÷ effort ÷ risk)

### O1 — Per-rod support cropping in `cyclic3d`  *(5–20×, ~50 lines, low risk)*

The 3D convolution only needs the result **at the ROI pixels** (the 1.5 µm
masks around each rod) — everything else is computed and thrown away. Convolve
each rod on its own `(rod support) ⊕ (PSF support)` sub-box and place the ROI
columns back. Cost becomes ∝ Σ rod-support volume instead of nseg × box volume.

- **XY cropping is exact** (same arithmetic, no boundary) — with two measured
  corrections, see §4: the sub-box half-width is `2·r_obj + r_roi + guard`
  (NOT `rod ⊕ PSF support`; the kernel is *cyclic-cropped*, never truncated),
  and the PSF array must be **cyclic-cropped from the full-box build**, not
  rebuilt at sub-grid size (the PSF build is grid-size dependent).
- **Z cropping needs the 99%-energy guard band** (the margin rule from
  `plans/manual-structure-selection.md` §2.4) because the convolution is
  *cyclic* in z: a rod's tail at the box edge wraps around and lands on the
  opposite plane. The 2 µm fixed margin used in the weekend runs is the
  suspected cause of the region3 bound-pinning (box 18 µm tall needs ~7 µm).
- **Risk: low** — identical math; gate with an equivalence test
  (support-cropped column vs full-box column at the rod's OWN ROI pixels,
  measured 1e-15 achievable, §4). Cross-rod far-tail entries need a model
  decision (§4.4) — cluster-grouped sub-boxes keep them exactly.
- **Does NOT change statistics** — unmarked neighbour bacteria still bias the
  fit; this is a speed fix, not a correctness fix.

### O2 — z-candidates from ONE FFT via phase ramp  *(~5×, ~30 lines, low risk)*

`np.roll(obj, dz)` per candidate currently costs a full 3D FFT each time. A
z-shift is a phase ramp in Fourier space: FFT the object **once**, multiply by
`exp(2πi k_z dz / N_z)`, IFFT per candidate. Kills the `nzcand` factor.
Gate: equivalence vs the `np.roll` path.

### O3 — Parallelize  *(2–6× per core, ~20–50 lines, low risk)*

Three independent levels:
- **Multi-starts** — embarrassingly parallel (N workers, trivial).
- **Jacobian columns** — the `npar+1` finite-difference evaluations of
  `least_squares` are independent (natural ~5-way for 4 params).
- **Within one evaluation** — the `nseg × nzcand` convolutions are independent
  (with O1 each is small).

Gotchas: dynamic scheduling for uneven rod supports; PSF volume duplication
(use `fork` COW or a shared read-only array); **determinism** — pin
`OMP_NUM_THREADS=1` per worker, reduce in fixed order, and add a test that the
parallel result matches the serial one bitwise (or to 1e-15) so reproducibility
isn't silently lost.

### O4 — Coarser search / cheaper eval, as flags  *(3–8×, small, low risk)*

All as **opt-in flags with a passing equivalence test**, never defaults:
- Coarse-then-refine z: search {0,±1} first, refine around the winner.
- PSF grid G = 256 (resolve the Airy core) instead of box size — 8× fewer
  voxels per 2D FFT; validate coefficients shift < 0.005 waves once.
- Warm start: initialize coefficients from the nearest already-fitted region
  (the field is smooth — physically justified) and run 1 start.
- 2× data downsampling for a first-pass estimate, full-res refinement.

### O5 — GPU  *(10–50× only on large boxes, large effort, higher risk — defer)*

- **cuFFT part: safe.** Same math, faster; batch all (rod × z-cand) FFTs into
  one batched call (small kernels are launch-overhead-bound, so batching is
  mandatory — and O1+O3 make the batch natural).
- **The NNLS is the real change.** scipy's Lawson–Hanson is an *exact
  sequential active-set* solver; GPU replacements are iterative first-order
  (projected gradient / FISTA) that converge to the same convex optimum *within
  a tolerance*. That residual error feeds the outer **non-convex** loop, so a
  tiny inner difference can flip which local minimum the whole fit lands in —
  especially in our ill-conditioned clusters (why we have multi-start).
- **Equivalence gate (required before any `--solver=gpu` flag):** run both
  solvers on ≥10 synthetic + 3 real clusters across all starts; require
  max|ΔZ| < 0.01 waves and identical flag/exclusion decisions.
- Not worth it for cluster-sized boxes (small kernels); worth it for whole-field
  fits, which only make sense with the field-model work — so defer.

## 2. Composition & order

O1 is the foundation: it makes each rod's work small *and* independent, which is
exactly what makes O3's parallelism and O5's batching efficient. Recommended
order: **O1 → O2 → O3 → O4 → (O5 only if whole-field fits become routine).**
Cumulatively O1–O4 give roughly **50–200×** with no new dependency.

## 2.5 Open design question — *what is one fit?* (field-varying aberration)

New constraint from the maintainer (2026-11-02): **the aberration differs
across the field** (astigmatism varies with image location). That invalidates
two implicit assumptions of the current tool:

1. **One shared Z5–Z8 per input box.** `fit_3d` already assumes this; a box
   spanning several aberration zones returns a meaningless average. Caveat:
   the week-end per-region results (Z5 from +0.32 to −0.50) *look* like field
   variation, but region3 also pinned at the ±0.5 bound, so today's numbers
   cannot yet separate "field variation" from "unstable fit".
2. **One fit over the whole field.** 542 Mvox × ~530 s/Mvox ≈ **80 h** *and*
   it would fit one constant aberration to a varying field — so a giant box is
   wrong for physics reasons, not only cost.

Consequence that must be decided **before** any speed-up is designed:

- **Crop grid ≠ arbitrary.** If Z(x,y) varies, the output is a *field map*
  Z(x,y), and the crops should tile the field at a spacing set by the
  aberration's spatial correlation length, with each tile's rods spread over
  that tile (not clustered at its centre). Tile size then has two independent
  bounds: **statistics** (enough rods × orientations to identify 4 modes) and
  **stationarity** (the aberration is ~constant inside a tile).
- Measuring the correlation length is a cheap, independent experiment:
  fit on overlapping small tiles along x of the full channel, then look at how
  Z5/Z6/Z7/Z8 decorrelate with distance. That single experiment answers "how
  big may a tile be?" and "is the field really varying?" at once.
- **Candidates for the tile geometry** (to be measured, not guessed):
  - XY tile: 96–160 px (11–18 µm), i.e. ~5–8× the Airy radius; z: full
    stack around the rods (needed for defocus identifiability), guard band
    **per rod**, not per tile.
  - Z window: must be several × the depth-of-field, and the 99 %-energy
    guard band must be computed per rod (the current shared bbox margin is
    what produced the 893² box for two rods).

Until the correlation length is known, no crop size should be committed to.

**Prerequisite (2026-11-03/04):** a stable fit. The region3 99 %-margin
re-run (correct z, 2026-11-03) pinned at −0.5 again → the guard-band
hypothesis is dead; the corrected z field-test diagnostic (±1.5 bound)
is pinned too (C seeds identical, 2026-11-04). Crop/tile geometry can be
decided only once a 9-rod fit comes off the bound
(`FIELDTEST_results_one_set_vs_Zmap.md` §8).

## 3. Discipline (applies to every option)

- Every "cheaper" variant is a **flag**, calibrated once against the full
  model: same cluster, both paths, require |ΔZ| < 0.01 waves, then the fast
  path is admissible. (The region3 bound-pinning is the warning: an unverified
  shortcut became silently-wrong results.)
- Performance changes land as `perf:` commits on a feature branch with the
  equivalence test in the suite.
- None of this changes the *statistics* of the fit (orientation coverage,
  isolation, guard band) — those live in `plans/manual-structure-selection.md`.

## 4. O1 implementation contract (measured 2026-11-16, `per_rod_probe.py`)

Prototype probe on synthetic confocal clusters (commit 74693c1, pixel
0.1135 µm/px, confocal model, 7 z-candidates). Numbers below are measured,
not assumed.

### 4.1 Crop-size rule (near field — exact)

- Sub-box half-width **`half ≥ 2·r_obj_half + r_roi + guard`**, with
  `r_obj_half` the rod's actual half-extent from the rendered object bounding
  box (≈ 25 px for a 45 px rod), `r_roi` = 1.5 µm ≈ 14 px, `guard` = 4 px.
  The object support appears on BOTH sides of every displacement
  `x − u` read at the ROI pixels — hence `2·r_obj`, not `r_obj`.
  Measured: **machine-exact (rel ≤ 1.2e-15) at half = 68 px** on a 512² box,
  for all rods including a deliberately close 15 px pair with overlapping
  ROIs. The earlier `rod ⊕ PSF support` (≈ 169 px) was derived wrongly and
  is 5× larger than needed.
- **Why not r_psf:** the kernel is never truncated. The full-box PSF array is
  *cyclic-cropped* to `±half` (rows/cols `[0..half] ∪ [G−half..G−1]`), which
  preserves the exact kernel values at every displacement the ROI pixels can
  read. The PSF's 99 %-energy radius (corrected 2026-11-17 — the probe's
  first radius metric broadcast silently to (m, n, n) and returned garbage
  129 px) is **14 px (1.6 µm) at ±6 µm volume defocus, 24 px at ±10.5 µm**
  (confocal, per-plane-renormalized) — it bounds only the rod-CLUSTERING
  reach and a possible future Z-crop, not the XY sub-box half-width.
- **Do NOT rebuild the PSF at sub-grid size.** The PSF build is grid-size
  dependent (pupil grid inscribed per `grid_size`): measured relative plateau
  ~1e-3 vs the full-box kernel, at every radius. Cyclic-cropping the full-box
  PSF is bit-identical and cheap (one gather per eval).
- **Z axis: keep the full box height initially** (43 planes on region5-like
  geometry is a small factor; the Z-crop and its guard band are a separate,
  later, separately-gated change).
- Sub-boxes must be **square** (pupil-grid convention), re-padded if clamped
  at the box edge.

### 4.2 Close rods produce identical results

Rod pairs 15–18 px apart with **overlapping ROIs** were tested explicitly:
their near-field columns match the full-box path to ≤ 1.2e-15. Closeness does
not break per-rod cropping — the NNLS re-couples rods through the shared ROI
pixels after the columns are gathered, exactly as before. Overlapping
sub-boxes are harmless (duplicate work only, never an error).

### 4.3 The far-field tail is the one real model decision

A rod's blurred column is NOT zero at the *other* rods' ROI pixels: the
defocused halo contributes up to **3.3e-3 of the column scale** at 100–200 px
separations (measured). The current full-box code includes these entries;
a naive per-rod sub-box zeroes them. Two admissible policies:

- **(a) Cluster-grouped sub-boxes (recommended default, exact):** group rods
  whose `(ROI ⊕ r_obj)` regions come within kernel reach; each group is
  convolved on ONE sub-box spanning the group's whole ROI union ⊕ r_obj.
  Within the group everything is exact, machine-precision — measured.
  A tight cluster (region3-like) then has one box ≈ its union bbox (small win
  over today, but drops the fixed 18 px margin and square-pad waste); a
  spread pair (region5-like, 800 px apart) becomes two independent boxes
  (large win).
- **(b) Cross-group truncation (opt-in flag, model change):** zero the far
  tail. Effect at other rods' ROI pixels ≤ 3.3e-3 of column scale — admissible
  ONLY after the §3 statistical gate (max |ΔZ| < 0.01 waves, identical
  bound-pinning flags) on ≥ 10 synthetic + 3 real clusters.

  **Correction (2026-11-17), found while implementing:** for rods separated
  by MORE than half the padded box, the current full-box code's cross-rod
  entries are not real signal at all — they are the *cyclic-wrap artifact*
  (the FFT kernel read at `separation mod box`, e.g. 800 px apart on an 893²
  box reads the kernel at 93 px ≈ 0.3 % of scale). Zeroing those entries is
  therefore arguably a **correction**, not an approximation — but it still
  changes fit outputs, so it stays behind the flag and the gate. Policy
  naming: `bank_policy="full_box"` reproduces current behavior bit-exactly;
  `bank_policy="cluster"` = grouped sub-boxes + cross-group zeros. Default
  stays `full_box` until the real-cluster calibration passes.

### 4.4 Measured speedup

Prototype probe (2026-11-16, 43×551×551, 2 rods 800 px apart, z full):
full-box bank 16.7 s → sub-box 1.81 s = **9×**.

Production code path (2026-11-17, nodeasb01g, idle, commit a179b62,
`scripts_extras/bank_bench_server.py`):

| geometry | full_box bank | cluster bank | speedup | near-field |
|---|---|---|---|---|
| region5-like: 43×893², 2 rods ~800 px apart | 90.7 s | 2.46 s | **36.8×** | 8.7e-16 |
| region3-like: 36×393², 7 rods ≤ ~200 px apart | 53.5 s | 47.3 s | 1.1× | bit-identical (fallback) |

The tight cluster merges into one group whose sub-box spans the box → the
code falls back to the full-box conv per group (identical math, no win).
Tight clusters need O2 (phase-ramp z-candidates) and O3 (parallelism); spread
clusters — the expensive weekend cases — get the bank win now. The PSF-volume
build is unchanged in both policies (9.4 s at 893²), as is the NNLS.

## 5. Solver policy — CPU and GPU run the SAME solver

**Lawson–Hanson (`scipy.optimize.nnls`) is the solver on both paths.** The
GPU accelerates the batched FFTs / PSF build / matrix assembly only; the
assembled matrix is transferred to the host once per eval and the NNLS +
outer `least_squares` run on CPU.

- **Why:** Lawson–Hanson is an exact, sequential active-set method — it does
  not vectorize to a GPU, and replacing it with an iterative solver
  (projected gradient / FISTA) changes returned solutions within solver
  tolerance, which feeds the outer non-convex loop (the region3 lesson).
  At `nseg + nz ≈ 43` columns the NNLS is ~7 % of eval cost (measured
  2026-10-30) — there is nothing meaningful to win.
- **Transfer cost is negligible:** region3-size A is ~250 MB float64 per eval
  → ~25 ms at 10 GB/s, against seconds of FFT time saved.
- **Numerics:** with the same solver, GPU vs CPU results differ only at FFT
  roundoff (~1e-15, different transform sizes/libraries). The §3 gate still
  applies (max |ΔZ| < 0.01 waves + identical pinning decisions), but no
  solver-tolerance term is introduced.
- **An iterative GPU NNLS is out of scope** unless post-O1–O3 profiling shows
  NNLS dominating. If it is ever added: separate opt-in flag, warm start from
  the previous eval's solution, strict §3 gate, and it must never be the
  default.
- **Device policy:** `--device auto|cpu|gpu`, default `auto`. `auto` = CuPy
  importable AND a CUDA device present AND the estimated footprint
  (batches × nz × sub² × 16 B × safety 3) fits in free VRAM; otherwise fall
  back to CPU **with a loud log line**, never silently. Result JSONs record
  device, solver, batch size, and the commit hash.
