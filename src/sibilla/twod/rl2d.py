"""Damped 2D Richardson–Lucy deconvolution (spec §7).

The estimator solves ``obs ≈ psf ⊛ object + background`` per tile with a
damped multiplicative update:

    x ← x · ((obs / (psf ⊛ x + bg)) ⊛ psfᵀ)          (RL step)
    x ← (1 - alpha) · x + alpha · x_new               (Landweber damping)

Two rules are load-bearing for the physics and pinned by the test suite:

- **The background is a CONSTANT in the denominator, never deconvolved.**
  Subtracting the pedestal first and deconvolving the residual amplifies
  pedestal shot noise into plausible-looking texture — the classic RL
  failure mode. Here ``bg`` enters the ratio as an additive constant and
  the caller supplies it (e.g. from
  :func:`sibilla.twod.flatfield.estimate_empty_background`); it is never
  estimated inside :func:`rl2d`.
- **Output stays ≥ 0.** The estimate is clipped at 0 after every
  iteration.

Convolutions are circular FFTs (:func:`fft_convolve2`), so the PSF must
have the same shape as the image with its peak at the array centre —
exactly the layout :func:`sibilla.twod.psf2d.psf2d` returns. The
back-projection uses the conjugate-FFT kernel — the exact adjoint of the
circular blur operator. The tempting flip ``p[::-1, ::-1]`` is not the
adjoint: for the even-size, centre-rolled kernels :func:`psf2d` returns,
the flip differs from the true adjoint by a 1-px roll for ANY centred
kernel (adjoint dot-product test: flip 16421.4 vs conjugate 16419.9), so
the conjugate form is required.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import fft as sp_fft

__all__ = ["RlResult", "rl2d", "fft_convolve2"]

_DENOM_EPS = 1e-12  # floor on blur + bg, same convention as the 3D RL


@dataclass
class RlResult:
    """Deconvolution result: ``image``, iterations applied, plateau flag."""

    image: np.ndarray
    iterations: int
    converged: bool


def fft_convolve2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Circular 2D convolution of two equal-shape arrays via FFT.

    Real part of ``ifft2(fft2(a) · fft2(b))``; both arrays must have the
    same shape. For a sum-normalized kernel this preserves the input sum.
    """
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if a.shape != b.shape:
        raise ValueError(
            f"fft_convolve2 needs equal shapes, got {a.shape} and {b.shape}"
        )
    return np.real(sp_fft.ifft2(sp_fft.fft2(a) * sp_fft.fft2(b)))


def rl2d(
    image: np.ndarray,
    psf: np.ndarray,
    background: float = 0.0,
    n_iter: int = 30,
    alpha: float = 0.5,
    metric=None,
    tol: float = 0.02,
    patience: int = 5,
) -> RlResult:
    """Damped 2D Richardson–Lucy deconvolution of one tile.

    image : 2D observed tile (same shape as ``psf``); the estimate is
        initialized from the (non-negative part of the) observation itself.
    psf : 2D energy-normalized PSF, peak at the array centre, same shape
        as ``image`` (as produced by :func:`sibilla.twod.psf2d.psf2d`).
        Re-normalized once by ``psf.sum()`` so an unnormalized kernel is
        tolerated.
    background : additive pedestal ``bg`` handed to RL as a CONSTANT in the
        ratio denominator. Caller-supplied (never estimated here) and never
        subtracted from ``image`` — see the module docstring for why.
    n_iter : maximum number of RL iterations.
    alpha : Landweber damping in ``(0, 1]``; ``1`` is plain RL, the default
        ``0.5`` halves each multiplicative step.
    metric : optional callable ``metric(x) -> float`` evaluated on the
        current estimate after each iteration; HIGHER is better (e.g. a
        recovery-quality score). Early stopping only applies when it is
        given.
    tol, patience : stop with ``converged=True`` once the metric has failed
        to improve by more than ``tol`` (relative to its previous value)
        for ``patience`` consecutive iterations.

    Returns an :class:`RlResult`; ``converged`` is False when ``n_iter`` is
    exhausted (or no ``metric`` is given) without a plateau.
    """
    obs = np.asarray(image, dtype=np.float32)
    p = np.asarray(psf, dtype=np.float32)
    if obs.ndim != 2 or p.ndim != 2:
        raise ValueError("image and psf must be 2D arrays")
    if obs.shape != p.shape:
        raise ValueError(
            f"image shape {obs.shape} != psf shape {p.shape} "
            "(circular FFT convolution requires equal shapes)"
        )
    if not 0.0 < alpha <= 1.0:
        raise ValueError(f"alpha must be in (0, 1], got {alpha}")
    if background < 0.0:
        raise ValueError(
            "background is an additive pedestal and must be >= 0, "
            f"got {background}"
        )
    energy = p.sum()
    if not energy > 0.0:
        raise ValueError("psf must have positive energy")
    p = p / energy  # energy-normalized once
    p_adj = np.real(sp_fft.ifft2(np.conj(sp_fft.fft2(p))))  # exact adjoint of the blur op
    bg = float(background)

    x = np.maximum(obs, 0.0)
    converged = False
    iterations = 0
    prev_metric: float | None = None
    stall = 0
    for i in range(1, n_iter + 1):
        blur = fft_convolve2(x, p)
        ratio = obs / (blur + bg + _DENOM_EPS)
        x_new = x * fft_convolve2(ratio, p_adj)
        x = (1.0 - alpha) * x + alpha * x_new
        np.maximum(x, 0.0, out=x)  # output stays >= 0
        iterations = i
        if metric is not None:
            m = float(metric(x))
            if prev_metric is not None and not (
                m - prev_metric > tol * abs(prev_metric)
            ):
                stall += 1
                if stall >= patience:
                    converged = True
                    break
            else:
                stall = 0
            prev_metric = m
    return RlResult(image=x, iterations=iterations, converged=converged)
