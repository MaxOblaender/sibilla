"""Evaluation metrics (plan §5: metrics.py).

- per-mode coefficient errors (fit - truth, RMS waves)
- wavefront RMS error: ||σ_fit - σ_truth|| (RMS over the unit pupil of the
  wavefront difference, in waves)
- residual RMS in ROI (photons/sqrt(px) scale)
- optional Richardson–Lucy deconvolution of the in-focus plane and
  apparent tubule diameter (FWHM of a radial cross-section profile),
  which ties the PSF estimate to the observable blur (plan §4 caveat 6).

metrics.py reads truth.json — it is the evaluation-only consumer of the
data contract (plan §4).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .fit import FitResult
from .psf import PupilConfig, build_psf_stack
from .zernike import noll_basis, pupil_grid


@dataclass
class EvalResult:
    truth: dict[int, float]
    fitted: dict[int, float]
    per_mode_err: dict[int, float]            # fit - truth, RMS waves
    rms_err_per_mode: dict[int, float]        # |err|
    wavefront_rms_error: float                # waves
    max_abs_err: float
    z_offset_um: float
    background: float
    amplitudes: np.ndarray = field(repr=False, default=None)
    roi_residual_rms: float = float("nan")


def _wavefront_rms(coeffs_a: dict[int, float], coeffs_b: dict[int, float],
                   grid: int = 256) -> float:
    """RMS (waves) of the wavefront difference on the unit pupil.

    W_a - W_b = Σ_j (a_j - b_j) Z_j;  RMS = sqrt(<(W_a-W_b)²>) over the pupil.
    With RMS-normalized orthonormal modes this equals sqrt(Σ_j (a_j-b_j)²),
    but we compute it numerically to guard against basis bugs.
    """
    diff = {}
    for j in set(coeffs_a) | set(coeffs_b):
        d = coeffs_a.get(j, 0.0) - coeffs_b.get(j, 0.0)
        if abs(d) > 0:
            diff[j] = d
    if not diff:
        return 0.0
    rho, theta, mask = pupil_grid(grid)
    Z = noll_basis(list(diff.keys()), rho, theta)
    w = (Z * np.array([diff[j] for j in diff.keys()])[:, None, None]).sum(axis=0)
    w = np.where(mask, w, 0.0)
    return float(np.sqrt((w ** 2).mean()))


def evaluate(fit_res: FitResult, truth: dict[int, float],
             modes: tuple[int, ...]) -> EvalResult:
    truth = {int(j): v for j, v in truth.items()}   # JSON keys arrive as str
    fitted = {j: fit_res.coeffs.get(j, 0.0) for j in modes}
    err = {j: fitted[j] - truth.get(j, 0.0) for j in modes}
    return EvalResult(
        truth={j: truth.get(j, 0.0) for j in modes},
        fitted=fitted,
        per_mode_err=err,
        rms_err_per_mode={j: abs(err[j]) for j in modes},
        wavefront_rms_error=_wavefront_rms(truth, fitted),
        max_abs_err=float(max(abs(v) for v in err.values())) if err else 0.0,
        z_offset_um=fit_res.z_offset_um,
        background=fit_res.background,
        amplitudes=fit_res.amplitudes,
    )


def roi_residual_rms(fit_res: FitResult, masks: list[np.ndarray]) -> float:
    """RMS of the raw residuals restricted to the ROI masks."""
    r = fit_res.residuals
    sel = np.zeros(r.size, dtype=bool)
    for k in range(r.shape[0]):
        m = r.shape[0]  # planes
        flat = np.zeros(r.shape[1] * r.shape[2], dtype=bool)
        for mask in masks:
            flat |= mask
        sel[k * r.shape[1] * r.shape[2] : (k + 1) * r.shape[1] * r.shape[2]] = flat
    return float(np.sqrt(np.mean(r.reshape(-1)[sel] ** 2)))


def richardson_lucy(image: np.ndarray, psf: np.ndarray,
                    n_iter: int = 30, eps: float = 1e-12) -> np.ndarray:
    """Richardson–Lucy deconvolution of a single plane (non-negative)."""
    from scipy import fft as sp_fft
    image = np.maximum(np.asarray(image, float), 0.0)
    psf = np.asarray(psf, float)
    psf = psf / psf.sum()
    m, n = image.shape
    M, N = 2 * m, 2 * n
    F_psf = np.zeros((M, N), complex)
    F_psf[:m, :n] = sp_fft.fft2(psf)
    F_psf = sp_fft.fftshift(F_psf) * 0  # keep unshifted for fftconvolve-free path
    # use fftconvolve for blurring, fft for the quotient
    from scipy.signal import fftconvolve
    est = np.full_like(image, image.mean() if image.size else 0.0)
    for _ in range(n_iter):
        blur = fftconvolve(est, psf, mode="same") + eps
        ratio = image / blur
        est = est * fftconvolve(ratio, psf[::-1, ::-1], mode="same")
        est = np.maximum(est, 0.0)
    return est


def apparent_diameter_um(image: np.ndarray, pixel_size: float,
                          center: tuple[float, float] | None = None,
                          direction: tuple[float, float] | None = None,
                          profile_len: int = 41) -> float:
    """Apparent diameter (µm) from the FWHM of a 1D radial cross-section.

    Takes a perpendicular profile through the tubule center (direction =
    segment direction, perpendicular profile is the diameter). Returns the
    FWHM of the smoothed profile in µm.
    """
    m, n = image.shape
    if center is None:
        iy, ix = np.unravel_index(np.argmax(image), image.shape)
        center = (float(ix), float(iy))
    if direction is None:
        direction = (0.0, 1.0)
    dx, dy = direction
    norm = np.hypot(dx, dy)
    dx, dy = dx / norm, dy / norm
    # perpendicular
    px, py = -dy, dx
    cs = np.linspace(-profile_len / 2, profile_len / 2, profile_len) * pixel_size
    # sample on a fine grid with bilinear-ish (nearest on 100nm grid is ok)
    vals = []
    for s in cs:
        x = center[0] + px * s / pixel_size
        y = center[1] + py * s / pixel_size
        x = int(round(x))
        y = int(round(y))
        if 0 <= x < n and 0 <= y < m:
            vals.append(image[y, x])
        else:
            vals.append(0.0)
    prof = np.asarray(vals)
    prof = prof / prof.max() if prof.max() > 0 else prof
    peak = prof.max()
    half = 0.5 * peak
    above = prof >= half
    if above.sum() < 2:
        return float("nan")
    idx = np.where(above)[0]
    width = (idx.max() - idx.min()) * pixel_size
    return float(width)
