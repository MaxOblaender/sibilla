"""Ensemble Zernike fitter (plan §5 "Fitter structure").

Design (ensemble principle):

- OUTER: nonlinear least squares over the 4 shared Zernike coefficients
  (Z5, Z6, Z7, Z8) using `scipy.optimize.least_squares` (numerical
  Jacobian), bounds ±0.5 RMS waves, multi-start (zero + k random).
- INNER (given a PSF candidate): per-segment amplitudes (one scalar per
  tubule) and ONE shared background, solved exactly by non-negative
  least squares (`scipy.optimize.nnls`) — variable projection: for a
  fixed PSF the model is linear in (amp_s, background).
- The shared z-offset nuisance is searched on a small grid
  {-200, -100, 0, +100, +200} nm (plan §5); the outer multi-start
  optimization is re-run at each grid point and the overall best
  (lowest cost) is kept.
- Residuals are evaluated only inside per-segment ROI masks (skeleton
  dilated by ~1.5 µm, plan §5). A robust `soft_l1` loss is available.

Model per plane j and segment s:
    pred_j = Σ_s amp_s · B_nom · (O_s ⊛ PSF_j(z0)) + background
where O_s is the unit-photon tubule object (scene.render_tubule_unit,
sum == 1), B_nom the nominal per-tubule photon budget (so amp_s ≈ 1 at
the true scene), PSF_j(z0) the scalar PSF evaluated at defocus
(z_planes_um[j] + z0), and z0 the shared z-offset.

The fitter NEVER reads truth.json (data contract, plan §4): it consumes
only the image stack + meta.json fields (pixel size, z-step, λ, NA, n,
diameter, skeleton, z-planes, modes, photon scale, B_nom).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from scipy import ndimage
from scipy.optimize import least_squares, nnls

from .defaults import REAL_MODES, Z_CANDIDATE_GRID_UM, Z_OFFSET_GRID_NM
from .forward import FastBlur, convolve_plane, cyclic3d, psf3d_cyclic
from .psf import PupilConfig, build_psf_stack
from .scene import render_tubule_unit

# Shared z-offset nuisance grid (defaults.py is the single source of truth;
# a module-level tuple is safe as an immutable dataclass default).
_Z_OFFSET_GRID_DEFAULT: tuple[float, ...] = tuple(Z_OFFSET_GRID_NM)


@dataclass
class FitInput:
    """Everything the fitter needs (generator-side inputs only)."""
    data: np.ndarray                        # (Nz, M, N) float, photon units
    config: PupilConfig
    skeleton: np.ndarray                    # (S, 4) [x0, y0, x1, y1] px
    z_planes_um: np.ndarray                 # (Nz,) detector plane defocus, µm
    nominal_budget: float                   # B_nom, photons per tubule at amp=1
    fluorescent_diameter_um: float          # object-model anchor (from meta)
    modes: tuple[int, ...] = (5, 6, 7, 8)
    roi_radius_um: float = 1.5
    z_offset_grid_nm: Sequence[float] = _Z_OFFSET_GRID_DEFAULT
    loss: str = "linear"                    # "linear" | "soft_l1"
    n_multi_start: int = 4                  # random starts beyond the zero start
    start_range: float = 0.2                # random starts uniform in ±start_range
    max_nfev: int = 200
    rng_seed: int = 0
    supersample: int = 8                    # scene supersampling (for O_s render)
    fov_px: int = 256


@dataclass
class FitResult:
    coeffs: dict[int, float]                # fitted Zernike amplitudes (RMS waves)
    amplitudes: np.ndarray                  # (S,) per-segment amplitudes
    background: float                       # shared background, photons/px
    z_offset_um: float                      # selected shared z-offset, µm
    cost: float                             # 0.5·sum(residuals²) at the solution
    residuals: np.ndarray                   # (Nz, M, N) raw residuals at the solution
    history: list = field(default_factory=list)   # [(z0_nm, best_cost), ...]
    segment_z_um: np.ndarray | None = None  # per-segment z offsets (fit_3d)


def _circle_footprint(radius_px: int) -> np.ndarray:
    r = max(1, int(round(radius_px)))
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    return (xx * xx + yy * yy) <= r * r


def _build_roi_masks(skeleton: np.ndarray, shape: tuple[int, int],
                     radius_um: float, pixel_size: float) -> list[np.ndarray]:
    """Per-segment ROI: 1-px skeleton dilated to a disk of radius_um."""
    m, n = shape
    footprint = _circle_footprint(radius_um / pixel_size)
    masks = []
    for (x0, y0, x1, y1) in skeleton:
        mask = np.zeros((m, n), dtype=bool)
        _draw_line(mask, x0, y0, x1, y1)
        masks.append(ndimage.binary_dilation(mask, structure=footprint))
    return masks


def _draw_line(mask: np.ndarray, x0: float, y0: float, x1: float, y1: float) -> None:
    """Bresenham 1-px line into `mask` (x = column, y = row)."""
    x0, y0, x1, y1 = (int(round(v)) for v in (x0, y0, x1, y1))
    dx, dy = abs(x1 - x0), -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    m, n = mask.shape
    while True:
        if 0 <= x0 < n and 0 <= y0 < m:
            mask[y0, x0] = True
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy


def _segment_objects(fit_in: FitInput) -> list[np.ndarray]:
    """Unit-photon tubule objects O_s (sum == 1 each), at the meta diameter."""
    objs = []
    m, n = fit_in.data.shape[1:]
    from .defaults import DEFAULT_SCENE, SceneSpec
    spec = SceneSpec(
        fov_px=n,
        pixel_size_um=fit_in.config.pixel_size,
        fluorescent_diameter_um=fit_in.fluorescent_diameter_um,
        supersample=fit_in.supersample,
    )
    for (x0, y0, x1, y1) in fit_in.skeleton:
        objs.append(render_tubule_unit(x0, y0, x1, y1, spec))
    return objs


class _LinearModel:
    """Fixed-PSF linear model: build A once, solve NNLS per call.

    pred = A @ [amp_1..amp_S, b] over the ROI-masked pixels of all planes.
    """

    def __init__(self, data: np.ndarray, objs: list[np.ndarray],
                 psf: np.ndarray, masks: list[np.ndarray],
                 nominal_budget: float):
        nz, m, n = data.shape
        self.m, self.n = m, n
        self.nz = nz
        # pixel indices of the union of the ROI masks (flat y*n + x within a plane)
        idx_list = []
        for mask in masks:
            ys, xs = np.where(mask)
            idx_list.append(ys * n + xs)
        self.idx = (np.unique(np.concatenate(idx_list)) if idx_list
                    else np.array([0]))
        # row k*L:(k+1)*L of b and A corresponds to plane k, ROI pixel order
        plane_off = (np.arange(nz) * m * n)[:, None]
        self.flat = (self.idx[None, :] + plane_off).ravel()     # (nz*L,)
        self.b = data.reshape(-1)[self.flat]
        # column per segment: blurred object on every plane at masked pixels
        self.A = np.empty((nz * len(self.idx), len(objs) + 1))
        col = 0
        for o in objs:
            fb = FastBlur(o)
            for k in range(nz):
                img = fb.blur(psf[k])
                self.A[k * len(self.idx):(k + 1) * len(self.idx), col] = (
                    img.reshape(-1)[self.idx])
            col += 1
        self.A[:, col] = 1.0                      # shared background column
        self.nominal_budget = nominal_budget

    def roi(self, pred: np.ndarray) -> np.ndarray:
        """ROI-masked values of a full (nz, m, n) prediction, matching b."""
        return pred.reshape(-1)[self.flat]

    def solve(self) -> np.ndarray:
        # NNLS in PHOTON units: x = [photon_amplitude_s, background].
        # (A columns are unit-photon blurs; b is photons.)
        return nnls(self.A, self.b)[0]

    def predict(self, sol: np.ndarray) -> np.ndarray:
        amps_ph = sol[:-1]
        bg = sol[-1]
        out = np.zeros((self.nz, self.m, self.n))
        for k in range(self.nz):
            row = slice(k * len(self.idx), (k + 1) * len(self.idx))
            seg = self.A[row, :self.A.shape[1] - 1] @ amps_ph
            out[k].reshape(-1)[self.idx] = seg + bg
        return out


class _LinearModel3D:
    """Fixed-PSF linear model for 3D stacks: per-plane backgrounds +
    data-driven Poisson weights (spec §3 fit.py row).

    pred rows: ROI pixels of all planes; columns: one per segment plus ONE
    PER PLANE (background). Weights w_i = 1/sqrt(max(data_i, floor)) with
    floor = 5% of the median ROI datum — gain-invariant: scaling the data
    by g scales the weighted system's rows uniformly, so the argmin
    (Zernike-relevant part) is unchanged (Review Focus 1).
    """

    def __init__(self, data: np.ndarray, objs: list[np.ndarray],
                 psf: np.ndarray, masks: list[np.ndarray],
                 nominal_budget: float,
                 per_plane_bg: bool = True,
                 poisson_weights: bool = True):
        nz, m, n = data.shape
        self.m, self.n, self.nz = m, n, nz
        idx_list = []
        for mask in masks:
            ys, xs = np.where(mask)
            idx_list.append(ys * n + xs)
        self.idx = (np.unique(np.concatenate(idx_list)) if idx_list
                    else np.array([0]))
        plane_off = (np.arange(nz) * m * n)[:, None]
        self.flat = (self.idx[None, :] + plane_off).ravel()
        self.b = data.reshape(-1)[self.flat].astype(np.float64)
        nbg = nz if per_plane_bg else 1
        # zero-init: per-plane bg columns are 1.0 only within their plane's
        # rows; every other entry must be exactly 0 (np.empty leaves garbage)
        self.A = np.zeros((nz * len(self.idx), len(objs) + nbg))
        col = 0
        for o in objs:
            fb = FastBlur(o)
            for k in range(nz):
                img = fb.blur(psf[k])
                self.A[k * len(self.idx):(k + 1) * len(self.idx), col] = (
                    img.reshape(-1)[self.idx])
            col += 1
        if per_plane_bg:
            for k in range(nz):
                self.A[k * len(self.idx):(k + 1) * len(self.idx), col + k] = 1.0
        else:
            self.A[:, col] = 1.0
        # Poisson weights: w = 1/sqrt(max(b, floor)), floor = 5% median ROI
        if poisson_weights:
            floor = 0.05 * float(np.median(self.b))
            w = 1.0 / np.sqrt(np.maximum(self.b, floor))
        else:
            w = np.ones_like(self.b)
        self._w = w
        self.Aw = self.A * w[:, None]
        self.bw = self.b * w
        self.nominal_budget = nominal_budget
        self.nbg = nbg

    def solve(self) -> np.ndarray:
        from scipy.optimize import nnls
        return nnls(self.Aw, self.bw)[0]

    def predict(self, sol: np.ndarray) -> np.ndarray:
        out = np.zeros((self.nz, self.m, self.n))
        nseg = self.A.shape[1] - self.nbg
        amps = sol[:nseg]
        for k in range(self.nz):
            row = slice(k * len(self.idx), (k + 1) * len(self.idx))
            seg = self.A[row, :nseg] @ amps
            if self.nbg == self.nz:
                seg = seg + sol[nseg + k]
            else:
                seg = seg + sol[nseg]
            out[k].reshape(-1)[self.idx] = seg
        return out

    def roi(self, pred: np.ndarray) -> np.ndarray:
        return pred.reshape(-1)[self.flat]

    def cost(self, sol: np.ndarray) -> float:
        r = self.bw - self.Aw @ sol
        return float(0.5 * (r * r).sum())


def fit(fit_in: FitInput) -> FitResult:
    """Run the ensemble Zernike fit; return the best FitResult."""
    config = fit_in.config
    modes = fit_in.modes
    objs = _segment_objects(fit_in)
    masks = _build_roi_masks(fit_in.skeleton, fit_in.data.shape[1:],
                             fit_in.roi_radius_um, config.pixel_size)
    npar = len(modes)
    bounds = (-0.5, 0.5)

    rng = np.random.default_rng(fit_in.rng_seed)
    starts = [np.zeros(npar)]
    starts += [rng.uniform(-fit_in.start_range, fit_in.start_range, size=npar)
               for _ in range(fit_in.n_multi_start)]

    def build_model(z0_um: float, coeffs_vec: np.ndarray) -> _LinearModel:
        coeffs = {int(j): float(c) for j, c in zip(modes, coeffs_vec)}
        z_shifted = fit_in.z_planes_um + z0_um
        psf = build_psf_stack(config, coeffs, z_shifted).psf
        return _LinearModel(fit_in.data, objs, psf, masks, fit_in.nominal_budget)

    history: list[tuple[float, float]] = []
    best: FitResult | None = None
    for z0_nm in fit_in.z_offset_grid_nm:
        z0_um = z0_nm / 1000.0

        def residuals(c: np.ndarray, _z0: float = z0_um) -> np.ndarray:
            model = build_model(_z0, c)
            sol = model.solve()
            pred = model.predict(sol)
            return model.b - model.roi(pred)

        best_c = starts[0]
        best_cost = np.inf
        n_fail = 0
        for s in starts:
            try:
                res = least_squares(residuals, s, bounds=bounds,
                                    loss=fit_in.loss, max_nfev=fit_in.max_nfev)
            except Exception:
                # a failed start must not silently poison the result
                n_fail += 1
                continue
            if res.cost < best_cost:
                best_cost = res.cost
                best_c = res.x
        if best_cost == np.inf:
            raise RuntimeError(
                f"fit failed at z0={z0_nm} nm: all {len(starts)} starts failed "
                f"({n_fail} exceptions, incl. overflow/NaN in the linear solve)")
        history.append((z0_nm, float(best_cost)))
        if best is None or best_cost < best.cost:
            model = build_model(z0_um, best_c)
            sol = model.solve()
            pred = model.predict(sol)
            best = FitResult(
                coeffs={int(j): float(best_c[i]) for i, j in enumerate(modes)},
                # dimensionless amplitudes (sol is in photon units): amp_s = photons_s / B_nom
                amplitudes=(sol[:-1] / fit_in.nominal_budget).copy(),
                background=float(sol[-1]),
                z_offset_um=z0_um,
                cost=float(best_cost),
                residuals=fit_in.data - pred,
                history=list(history),
            )
    assert best is not None
    return best


@dataclass
class FitInput3D:
    """3D ensemble fit input (spec §3 fit.py row, real-data plan Task 6).

    data is (Nz, M, N) camera counts on a SQUARE grid (io.square_pad);
    config.grid_size must equal M == N. skeleton rows are
    [x0, y0, z0, x1, y1, z1] with z in PLANE units. z_planes_um are the
    ABSOLUTE plane z positions (µm).
    """
    data: np.ndarray
    config: PupilConfig
    skeleton: np.ndarray                       # (S, 6)
    z_planes_um: np.ndarray                    # (Nz,) absolute
    nominal_budget: float
    fluorescent_diameter_um: float
    modes: tuple[int, ...] = REAL_MODES
    roi_radius_um: float = 1.5
    z_candidate_um: Sequence[float] = Z_CANDIDATE_GRID_UM
    psf_model: str = "widefield"        # "widefield" | "confocal" (product)
    lambda_exc_nm: float = 493.0        # excitation λ (confocal model only)
    loss: str = "linear"
    n_multi_start: int = 1          # smooth 4-param problem; zero start only
    start_range: float = 0.2
    max_nfev: int = 200
    rng_seed: int = 0
    n_cd_sweeps: int = 2
    n_outer_sweeps: int = 3        # alternating (c | z) block iterations


def _capsule_objects(fit_in: "FitInput3D", z_off_plane: np.ndarray
                     ) -> list[np.ndarray]:
    """Unit-sum capsule objects, each shifted by its z-candidate (plane units).

    Returns (Nz, M, N) footprints rolled along z so the capsule sits at
    (segment z + candidate offset).
    """
    from .scene import CapsuleSpec, render_capsule_unit

    nz, m, n = fit_in.data.shape
    spec = CapsuleSpec(fov_px_y=m, fov_px_x=n, n_planes=nz,
                       pixel_size_um=fit_in.config.pixel_size,
                       z_step_um=float(np.median(np.diff(fit_in.z_planes_um)))
                       if len(fit_in.z_planes_um) > 1 else 0.5,
                       diameter_um=fit_in.fluorescent_diameter_um,
                       aa_width_px=1.0)
    objs = []
    for s, (x0, y0, z0, x1, y1, z1) in enumerate(fit_in.skeleton):
        dz = z_off_plane[s]
        obj = render_capsule_unit(x0, y0, z0 + dz, x1, y1, z1 + dz, spec)
        objs.append(obj)
    return objs


def _blur_columns(fit_in: "FitInput3D", objs: list[np.ndarray],
                  psf: np.ndarray, masks: list[np.ndarray],
                  model_cls=_LinearModel3D
                  ) -> "_LinearModel3D":
    """_LinearModel3D with per-PLANE blurs of per-plane footprints."""
    nz, m, n = fit_in.data.shape
    idx_list = []
    for mask in masks:
        ys, xs = np.where(mask)
        idx_list.append(ys * n + xs)
    idx = np.unique(np.concatenate(idx_list)) if idx_list else np.array([0])
    L = len(idx)
    plane_off = (np.arange(nz) * m * n)[:, None]
    flat = (idx[None, :] + plane_off).ravel()
    b = fit_in.data.reshape(-1)[flat].astype(np.float64)
    nseg = len(objs)
    A = np.zeros((nz * L, nseg + nz))
    for s, o in enumerate(objs):
        for k in range(nz):
            fbp = FastBlur(o[k])
            img = fbp.blur(psf[k])
            A[k * L:(k + 1) * L, s] = img.reshape(-1)[idx]
    for k in range(nz):
        A[k * L:(k + 1) * L, nseg + k] = 1.0
    floor = 0.05 * float(np.median(b))
    w = 1.0 / np.sqrt(np.maximum(b, floor))
    model = _LinearModel3D.__new__(_LinearModel3D)
    model.m, model.n, model.nz = m, n, nz
    model.idx = idx
    model.flat = flat
    model.b = b
    model.A = A
    model.Aw = A * w[:, None]
    model.bw = b * w
    model._w = w
    model.nominal_budget = fit_in.nominal_budget
    model.nbg = nz
    return model


def fit_3d(fit_in: FitInput3D) -> FitResult:
    """Ensemble 3D fit (spec §3 fit.py row) — physically-correct depth model.

    Each rod s (3D segment at plane z_s) is imaged with the PSF evaluated at
    ITS OWN depth: on data plane k the defocus is (k - z_s)·z_step. With the
    cyclic-z PSF volume this is image = Σ_s amp_s · (O_s ⊛3D PSF), i.e. the
    forward model is a 3D cyclic convolution per segment. A per-segment
    z-candidate d_s shifts the rod's footprint by d_s planes (np.roll),
    equivalently defocusing its PSF by d_s·z_step — so candidate search is
    cheap (one NNLS on precomputed blurred columns per candidate).

    OUTER: least_squares over the shared Zernike coeffs (bounds ±0.5,
    multi-start). INNER: per-segment z-candidate coordinate descent +
    weighted per-plane-background NNLS (variable projection).
    """
    from .scene import CapsuleSpec, render_capsule_unit

    config = fit_in.config
    modes = fit_in.modes
    nz, m, n = fit_in.data.shape
    z_steps = (np.diff(fit_in.z_planes_um) if len(fit_in.z_planes_um) > 1
               else np.array([0.5]))
    z_step = float(np.median(z_steps))
    nseg = len(fit_in.skeleton)
    z_plane_int = np.array([int(round(z / z_step))
                            for z in fit_in.z_candidate_um])
    z_mid = int(np.argmin(np.abs(z_plane_int)))   # index of the 0-offset candidate
    bounds = (-0.5, 0.5)
    rng = np.random.default_rng(fit_in.rng_seed)
    npar = len(modes)
    starts = [np.zeros(npar)]
    starts += [rng.uniform(-fit_in.start_range, fit_in.start_range, npar)
               for _ in range(fit_in.n_multi_start)]

    spec = CapsuleSpec(fov_px_y=m, fov_px_x=n, n_planes=nz,
                       pixel_size_um=config.pixel_size, z_step_um=z_step,
                       diameter_um=fit_in.fluorescent_diameter_um,
                       aa_width_px=1.0)

    # ROI pixel indices (shared across all evaluations)
    masks = _build_roi_masks(fit_in.skeleton[:, [0, 1, 3, 4]], (m, n),
                             fit_in.roi_radius_um, config.pixel_size)
    idx_list = []
    for mask in masks:
        ys, xs = np.where(mask)
        idx_list.append(ys * n + xs)
    idx = (np.unique(np.concatenate(idx_list)) if idx_list
           else np.array([0]))
    L = len(idx)
    plane_off = (np.arange(nz) * m * n)[:, None]
    flat = (idx[None, :] + plane_off).ravel()
    b = fit_in.data.reshape(-1)[flat].astype(np.float64)
    floor = 0.05 * float(np.median(b))
    w = 1.0 / np.sqrt(np.maximum(b, floor))

    def blurred_bank(coeffs_vec: np.ndarray) -> list[np.ndarray]:
        """For each segment and each z-candidate, the blurred (nz,L) column.

        Cost control: 3D cyclic convolution per segment per candidate (the
        plan's "precompute per-(segment × z-candidate) blurred stacks once").
        """
        coeffs = {int(j): float(c) for j, c in zip(modes, coeffs_vec)}
        psf = psf3d_cyclic(config, coeffs, nz, z_step,
                           model=fit_in.psf_model,
                           lambda_exc_um=fit_in.lambda_exc_nm / 1000.0)
        bank: list[np.ndarray] = []
        for s, seg in enumerate(fit_in.skeleton):
            obj = render_capsule_unit(seg[0], seg[1], seg[2],
                                      seg[3], seg[4], seg[5], spec)
            cols = []
            for dz in z_plane_int:
                o = obj if dz == 0 else np.roll(obj, dz, axis=0)
                col = cyclic3d(o, psf).reshape(nz, -1)[:, idx]
                cols.append(col)
            bank.append(cols)
        return bank

    def solve_at(bank: list[np.ndarray], z_sel: np.ndarray):
        """Weighted per-plane-bg NNLS with column s = bank[s][z_sel[s]].

        NOTE: A must be fully initialized (np.zeros). np.empty left the
        per-plane background columns uninitialized outside their own z-block
        (allocation garbage) — a day-long bug: nondeterministic residuals,
        garbage FD Jacobians, fits "succeeding" at the start point, tests
        passing on fresh zero pages and failing later. Verified 2026-10-20.
        """
        A = np.zeros((nz * L, nseg + nz))
        for s in range(nseg):
            A[:, s] = np.concatenate(bank[s][z_sel[s]])
        for k in range(nz):
            A[k * L:(k + 1) * L, nseg + k] = 1.0
        Aw = A * w[:, None]
        bw = b * w
        sol = nnls(Aw, bw)[0]
        cost = float(0.5 * ((bw - Aw @ sol) ** 2).sum())
        return A, sol, cost

    def coordinate_descent(bank: list[np.ndarray], z0idx: np.ndarray):
        """Greedy per-segment z search.

        `z_sel` holds INDICES into the candidate list (bank[s][i] is the
        blurred column for offset z_plane_int[i]). Each sweep tries every
        candidate index per segment and keeps the absolute best index —
        no offset/index mixing, no accumulation across sweeps.
        """
        z_sel = np.asarray(z0idx, dtype=int).copy()
        _, _, cost = solve_at(bank, z_sel)
        for _ in range(fit_in.n_cd_sweeps):
            for s in range(nseg):
                best_c, best_i = cost, int(z_sel[s])
                for i in range(len(z_plane_int)):
                    if i == best_i:
                        continue
                    trial = z_sel.copy()
                    trial[s] = i
                    _, _, c = solve_at(bank, trial)
                    if c < best_c - 1e-12:
                        best_c, best_i = c, i
                z_sel[s] = best_i
                cost = best_c
        A, sol, cost = solve_at(bank, z_sel)
        return A, z_sel, sol, cost

    best_overall: tuple[float, np.ndarray, np.ndarray, np.ndarray] | None = None
    for start in starts:
        # --- Alternating block optimization (2026-10-20 fix) ---
        # The old design ran coordinate descent INSIDE residuals(), making
        # the residual discontinuous in c (near-tied z candidates flip between
        # calls), so the finite-difference Jacobian was garbage and
        # least_squares stalled at the start point (returned c=0 for a
        # nonzero injected aberration; xtol termination). Now: fit c with z
        # FROZEN (residual is smooth in c -> valid Jacobian), then update z
        # by CD with c frozen, repeat. Same optimum, smooth outer problem.
        c_cur = np.asarray(start, dtype=float).copy()
        z_cur = np.full(nseg, z_mid, dtype=int)
        for _ in range(max(1, fit_in.n_outer_sweeps)):
            # residuals MUST depend on c: rebuild the bank from c (the old
            # version captured a prebuilt bank -> residual constant in c ->
            # Jacobian == 0 -> optimizer never moved off the start point).
            def residuals(c: np.ndarray) -> np.ndarray:
                bank_c = blurred_bank(c)
                A, sol, _ = solve_at(bank_c, z_cur)
                return (b - (A @ sol)) * w

            try:
                res = least_squares(residuals, c_cur, bounds=bounds,
                                    loss=fit_in.loss, max_nfev=fit_in.max_nfev)
                c_cur = res.x
            except Exception as exc:
                # NEVER silent: a bare `pass` here once hid a hard crash
                # (unpacking error) and the fit "succeeded" at the start point.
                import warnings
                warnings.warn(f"fit_3d: outer least_squares failed "
                              f"({exc!r}); keeping previous coefficients",
                              RuntimeWarning, stacklevel=2)
            bank = blurred_bank(c_cur)
            A, z_cur, sol, cost = coordinate_descent(
                bank, z_cur)          # warm-start: refine z from previous
        if best_overall is None or cost < best_overall[0]:
            best_overall = (cost, c_cur.copy(), z_cur.copy(), sol.copy())

    if best_overall is None:
        raise RuntimeError("fit_3d: all starts failed")

    cost, c_vec, z_sel, sol = best_overall
    pred_flat = np.zeros(nz * L)
    # recompute A once for the prediction (bank reuse avoided for clarity)
    bank = blurred_bank(c_vec)
    A = np.zeros((nz * L, nseg + nz))   # zeros: np.empty leaks garbage in bg cols
    for s in range(nseg):
        A[:, s] = np.concatenate(bank[s][z_sel[s]])
    for k in range(nz):
        A[k * L:(k + 1) * L, nseg + k] = 1.0
    pred_flat = A @ sol
    pred = np.zeros_like(fit_in.data)
    pred.reshape(-1)[flat] = pred_flat
    amps = sol[:nseg] / fit_in.nominal_budget
    # per-segment z offsets in µm (relative to the fitted segment position);
    # z_sel holds candidate INDICES, so map back through z_plane_int
    seg_z_um = z_plane_int[z_sel] * z_step
    return FitResult(
        coeffs={int(j): float(c_vec[i]) for i, j in enumerate(modes)},
        amplitudes=amps,
        background=float(np.mean(sol[nseg:])),
        z_offset_um=float(np.mean(seg_z_um)),
        cost=cost,
        residuals=fit_in.data - pred,
        history=[],
        segment_z_um=seg_z_um,
    )


def nominal_budget_from_data(stack: np.ndarray, masks: list[np.ndarray],
                             skeletons: np.ndarray | None = None) -> float:
    """Median over segments of the ROI sum at the segment's brightest plane.

    Counts-based B_nom: NNLS amplitudes become ~1 at typical rods; the
    weighted fit is gain-invariant regardless (Review Focus 1).
    """
    if skeletons is not None and len(skeletons):
        sums = []
        for sk in skeletons:
            x0, y0, x1, y1 = (int(round(v)) for v in sk[:4])
            n = stack.shape[2]
            m = stack.shape[1]
            # perp profile ROI: bounding box of the segment + 1 µm margin
            xa, xb = max(0, min(x0, x1) - 9), min(n, max(x0, x1) + 10)
            ya, yb = max(0, min(y0, y1) - 9), min(m, max(y0, y1) + 10)
            if xa >= xb or ya >= yb:
                continue
            box = stack[:, ya:yb, xa:xb]
            sums.append(float(box.max(axis=0).sum()))
        if sums:
            return float(np.median(sums))
    # fallback: sum of the brightest plane's mask union
    mip = stack.max(axis=0)
    sel = np.zeros(mip.shape, bool)
    for mask in masks:
        sel |= mask
    if not sel.any():
        sel = np.ones(mip.shape, bool)
    return float(mip[sel].sum())


def fit_from_real_files(stack: np.ndarray, meta: dict,
                        max_nfev: int = 200) -> FitResult:
    """Convenience: FitInput3D from a loaded stack + crop_meta dict."""
    config = PupilConfig(
        pixel_size=float(meta["pixel_size_um"]),
        lambda_um=float(meta["lambda_um"]),
        na=float(meta["na"]),
        n_im=float(meta["n_im"]),
        grid_size=stack.shape[1],
    )
    fit_in = FitInput3D(
        data=stack.astype(float),
        config=config,
        skeleton=np.asarray(meta["skeleton"], dtype=float),
        z_planes_um=np.asarray(meta["z_planes_um"], dtype=float),
        nominal_budget=float(meta["nominal_budget"]),
        fluorescent_diameter_um=float(meta["fluorescent_diameter_um"]),
        modes=tuple(meta.get("modes", REAL_MODES)),
        roi_radius_um=float(meta.get("roi_radius_um", 1.5)),
        z_candidate_um=tuple(meta.get("z_candidate_um",
                                      Z_CANDIDATE_GRID_UM)),
        max_nfev=max_nfev,
    )
    return fit_3d(fit_in)
    """Convenience: build FitInput from a loaded stack + meta.json dict."""
    config = PupilConfig(
        pixel_size=float(meta["pixel_size_um"]),
        lambda_um=float(meta["lambda_um"]),
        na=float(meta["na"]),
        n_im=float(meta["n_im"]),
        grid_size=meta["stack_shape"][1],
    )
    fit_in = FitInput(
        data=stack.astype(float),
        config=config,
        skeleton=np.asarray(meta["skeleton"], dtype=float),
        z_planes_um=np.asarray(meta["z_planes_um"], dtype=float),
        nominal_budget=float(meta["nominal_budget"]),
        fluorescent_diameter_um=float(meta["fluorescent_diameter_um"]),
        modes=tuple(meta.get("modes", [5, 6, 7, 8])),
        roi_radius_um=float(meta.get("roi_radius_um", 1.5)),
        max_nfev=max_nfev,
    )
    return fit(fit_in)
