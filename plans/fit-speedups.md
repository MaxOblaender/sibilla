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

## 3. Discipline (applies to every option)

- Every "cheaper" variant is a **flag**, calibrated once against the full
  model: same cluster, both paths, require |ΔZ| < 0.01 waves, then the fast
  path is admissible. (The region3 bound-pinning is the warning: an unverified
  shortcut became silently-wrong results.)
- Performance changes land as `perf:` commits on a feature branch with the
  equivalence test in the suite.
- None of this changes the *statistics* of the fit (orientation coverage,
  isolation, guard band) — those live in `plans/manual-structure-selection.md`.
