# Plan — Speeding up the Zernike fit (`fit_3d`)

Status: **design only, nothing implemented.** Measured numbers are from
2026-10-30 (local + nodeasb01) on the real AF488 channel, confocal model.

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

- **XY cropping is exact** (same arithmetic, no boundary).
- **Z cropping needs the 99%-energy guard band** (the margin rule from
  `plans/manual-structure-selection.md` §2.4) because the convolution is
  *cyclic* in z: a rod's tail at the box edge wraps around and lands on the
  opposite plane. The 2 µm fixed margin used in the weekend runs is the
  suspected cause of the region3 bound-pinning (box 18 µm tall needs ~7 µm).
- **Risk: low** — identical math; gate with an equivalence test
  (support-cropped column vs full-box column, `allclose` 1e-12).
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

## 3. Discipline (applies to every option)

- Every "cheaper" variant is a **flag**, calibrated once against the full
  model: same cluster, both paths, require |ΔZ| < 0.01 waves, then the fast
  path is admissible. (The region3 bound-pinning is the warning: an unverified
  shortcut became silently-wrong results.)
- Performance changes land as `perf:` commits on a feature branch with the
  equivalence test in the suite.
- None of this changes the *statistics* of the fit (orientation coverage,
  isolation, guard band) — those live in `plans/manual-structure-selection.md`.
