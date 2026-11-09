# Plan: `sibilla.twod` — 2D defocus estimation & deconvolution for PCLS tiles

**Status: DRAFT — design sketch, nothing implemented yet (2026-11-09).**
Branch: `feat/twod` (from `main` @ dff11ba). Motivating dataset:
`~/Documents/Data/data_for_PCLS_manuscript` (Keyence BZ-X710, PlanApo 4×,
NA 0.20, 1.887 µm/px, 8-bit; stitched mosaics + per-tile QC in
`~/Documents/git_repos/pcls/`).

## 1. Problem

PCLS well tiles are single 2D planes with **spatially varying defocus**:
the slice tilts/drifts, so one image contains an area of nice focus and
large blurred areas. The blur is dominated by defocus `dz` (autofocus was
OFF; fixed Z per acquisition), not by unknown aberrations. We want to
recover the recoverable tiles: estimate `dz` per tile/patch, then
Richardson–Lucy deconvolve with the corresponding PSF.

## 2. Relation to existing sibilla (3D rod machinery)

Reuse unchanged: `psf.py` (`PupilConfig`, `defocus_opd_waves`,
`build_psf_stack` — already accepts arbitrary `z_planes`), `forward.py`,
RL core concepts from `deconvolve.py`. NOT reused: `fit.py`, `scene.py`,
rod segmentation (3D/rod-specific).

New subpackage `src/sibilla/twod/`:

| module | content |
|---|---|
| `optics.py` | Keyence defaults (NA 0.20, 1.887 µm/px, per-channel λ), `PupilConfig` factory |
| `psf2d.py` | 2D PSF for a given `dz` (plane of `build_psf_stack`, or thickness-weighted sum over a z-range for thick slices) |
| `dz_estimate.py` | per-patch defocus estimators (see §3) |
| `flatfield.py` | per-tile background/illumination normalisation (must run BEFORE RL) |
| `rl2d.py` | 2D Richardson–Lucy (CPU + cupy GPU path for nodeasb01g) |
| `patchwise.py` | dz-map smoothing + patchwise deconvolution + feathered blending |

`psf.py`, `fit.py`, `deconvolve.py` (3D) are **not modified**. All 13
existing test files must keep passing.

## 3. Defocus estimation — the core idea (user's proposal, 2026-11-09)

A recoverable tile typically contains **an in-focus region next to blurred
regions**. The in-focus region is the reference; it anchors three things:

1. **Object-spectrum anchor.** Blurred spectrum = object spectrum ×
   |MTF(dz)|². The unknown object spectrum is estimated *empirically from
   the in-focus region of the same image* (same tissue!), and `dz` follows
   by fitting the spectral ratio against the model MTF family from
   `psf2d.py`. This cancels the main confound of blind spectral matching.
   Caveats to handle: local stationarity (use several reference patches,
   robust median ratio); noise floor at high f (fit below the noise knee);
   reference must be genuinely in focus (pick by sharpness).
2. **RL tuning.** Candidate (dz, n_iter) pairs are scored by how close the
   deconvolved region's texture statistics get to the in-focus reference.
3. **QC metric.** Post-deconvolution agreement with the in-focus reference
   is the objective "recovered" criterion for the manuscript.

Supplementary estimators (for tiles without point objects or texture):
conidium point-spread fit (sparse bright CH1 spots ≈ beads) and
neighbour interpolation (dz field is smooth — measured drift/tilt is
monotonic within wells).

Physics limits: NA 0.20 → lateral resolution ≈ 1 px at 1.887 µm/px
(near-Nyquist); depth of focus ≈ 15 µm; recoverable band ≈ 15–40 µm
defocus; beyond that high frequencies are destroyed (unrecoverable class).

## 4. Open questions (block spec finalisation)

- Deliverable: showcase wells (A) / measured recoverable-fraction claim (B)
  / reusable batch pipeline (C)?
- Zernike scope for pilot: Z5–Z8 = 0 + per-tile dz (recommended, defensible
  at NA 0.2 — cf. FIELDTEST §9/§10: fitted coefficients explained ~4 points
  over zero aberration) — or include a session-level Zernike fit?
- Pilot well choice (CH1 conidia + strong within-well focus gradient).
- Slice-thickness model: thin-section approximation per patch vs
  thickness-weighted PSF sum (PCLS thickness ~200 µm? ask supervisor).
- Out-of-focus haze from the slice bulk = the same unmodelled background
  that pinned the 3D fits (FIELDTEST §9) → background subtraction design.

## 5. Milestones (tentative)

1. `optics.py` + `psf2d.py` + unit tests (PSF width vs dz sanity).
2. `dz_estimate.py` with the in-focus-anchored spectral ratio; validate on
   one well: estimated dz map must reproduce the measured sharpness
   gradient (monotonicity test).
3. `rl2d.py` + `flatfield.py`; A/B on one blurry patch: before/after vs
   in-focus reference statistics.
4. `patchwise.py`; one full tile, then one full well; review figures.
5. GPU path + batch runner on nodeasb01g (4× A40, free 2026-11-09 —
   coordinate with jdoege26's JiPipe job, ~155 GB RAM, GPUs idle).
