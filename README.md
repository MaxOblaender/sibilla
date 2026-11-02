# SIBILLA (Python)

**Blind Zernike aberration recovery from 3D fluorescence microscopy stacks.**
Given a measured 3D image, SIBILLA determines the optical aberration state of the
imaging system — as **Zernike coefficients** (Z5–Z8: astigmatism and coma) — purely
from the data ("blind": no separate wavefront sensor), then builds the corresponding
PSF and deconvolves the volume.

This is the **Python successor to [Zoltan Cseresnyes' SIBILLA](../sibilla_zoltan/)**
(C#/.NET, 2008–2018). Where the original recovers Zernikes by matching a synthetic
point-spread *library* (beads) to a measured stack, this version fits the coefficients
directly against **rod-like line objects** (bacteria) in the data, with a physically
correct per-object depth model. See `plans/` for the full design history (local only,
not tracked).

## Install

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e ".[test,data]"
```

Core deps: numpy, scipy, tifffile, matplotlib. The `[data]` extra adds
czifile + scikit-image (needed to read Zeiss CZI and run segmentation).

## Usage

Three entry points. All take the imaging parameters (pixel size, z-step, λ, NA, n) —
defaults are pinned for the GoC-Exp4 water-immersion 40×/1.2 objective
(0.1135 µm px, 0.5 µm z, 517 nm, NA 1.2, n 1.33); see `src/sibilla/defaults.py`.

### 1. Fit Zernikes to one 3D crop

```bash
.venv/bin/python -m sibilla.fit_real_crop --crop <crop.tif> --out <dir> \
    --psf-model confocal --diameter-um 0.65
```
Reads the crop, auto-segments rod skeletons, fits Z5–Z8 (shared across rods, per-rod
depth + brightness), and writes `fit_report.json` + `psf.json` + a review figure.

### 2. Deconvolve a volume with the fitted PSF

```bash
.venv/bin/python -m sibilla.deconvolve --czi <file.czi> --channel 1 \
    --psf <dir>/psf.json --out <dir> --n-iter 6
```
3D Richardson–Lucy (CPU or GPU) with the fitted aberrated PSF. `--zero-coeffs` gives
the diffraction-limited reference; `--flat-z-kernel` switches the axial kernel.

### 3. Full real-data pipeline

```bash
.venv/bin/python -m sibilla.real_pipeline --czi <file.czi> --channel 1 --out <dir>
```
Auto-selects crops, runs the ensemble fit, and deconvolves.

### Workflow front-ends (`scripts/`)

`calibrate_single.py`, `crop_sanity.py`, `select_batch_crops_v2.py`,
`export_psf_for_dolphin.py`, `manual_crop.py` — helpers for the real-data workflow.

## Tests

```bash
.venv/bin/python -m pytest            # full suite (incl. real-data tests if data present)
.venv/bin/python -m pytest -q -x      # stop at first failure
```

Real-data tests auto-skip when the imaging files are absent; when present they run the
actual fit/deconvolution, so the full suite takes ~20 min on a CPU.

## Package layout

| File | Role |
|---|---|
| `zernike.py` | Noll orthonormal Zernike basis on a circular pupil grid |
| `psf.py` | coefficients → per-plane PSF (widefield + confocal product), exact defocus OPD, Strehl |
| `forward.py` | forward model: `FastBlur`, `cyclic3d`, `psf3d_cyclic`, plane/stack convolution |
| `fit.py` | the ensemble fitter: outer least-squares over Z5–Z8, inner NNLS (amplitude + per-plane bg) + per-rod z coordinate descent |
| `scene.py` | rod/capsule rendering (unit-sum footprints) |
| `segments.py` | semi-automatic rod skeleton extraction + flags (edge/saturation/sheared) |
| `io.py` | CZI + TIFF reading, NCC-verified crop extraction, square-pad |
| `defaults.py` | pinned real-data parameters (pixel, z-step, λ, NA, n, modes) |
| `fit_real_crop.py` | CLI: fit one crop → `psf.json` |
| `deconvolve.py` | CLI: 3D RL deconvolution with a `psf.json` |
| `real_pipeline.py` | CLI: crop selection → ensemble fit → deconvolve |
| `metrics.py`, `scene.py` helpers, `make_dataset.py`, `run_preflight.py` | validation/benchmark support |

## Conventions (pinned, enforced by tests)

- **Basis:** Noll indexing, orthonormal over the unit pupil, RMS-normalized.
  Wavefront RMS = `sqrt(Σ a_j²)`; coefficients in **RMS waves**;
  Strehl ≈ `exp(−(2πσ)²)`.
- **Modes:** Z5 (astig 45°), Z6 (astig 0/90°), Z7 (coma y), Z8 (coma x). Z1 (piston)
  is unobservable, Z2/Z3 (tilt) are degenerate with object position, Z4 (defocus) is
  handled by the per-rod depth nuisance — so they are not fitted.
- **PSF:** scalar; pupil phase = `Σ_j 2π a_j Z_j + exact geometric defocus OPD`,
  `PSF = |FFT(pupil)|²`, energy-normalized. Confocal = excitation × detection product
  at the two wavelengths (same Zernikes).
- **Depth:** per-rod free parameter (coordinate descent over a z-candidate window);
  the shared Zernikes are the measured quantity.

## Notes

- Data (CZI/TIF), generated results, and one-off analysis scripts live in this folder
  but are **not tracked** (see `.gitignore`) — only the code, tests, and workflow
  scripts are committed.
- `plans/manual-structure-selection.md` (local) documents the in-progress design for
  manual ROI-based rod selection.
