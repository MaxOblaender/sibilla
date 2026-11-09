"""Brightness handling for the 2D defocus pipeline (spec §6, three levels).

This module implements the first two, strictly separated levels:

- **L1 — per-tile normalization** (:func:`l1_normalize`): removes each tile's
  own background level and contrast scale so tiles become comparable before
  defocus estimation and RL. This kills the rectangular brightness steps
  between tiles that RL would otherwise amplify. Fluorescence channels are
  dark-background; the transmission channel (CH4) is bright-background and is
  inverted to *darkness* so both kinds map "empty" to 0 and "signal" to ~1.
- **L2 — cross-patch gain/offset matching** (:func:`l2_match`): fits
  ``target ≈ g·ref + c`` from low frequencies only and applies the *inverse*
  to the target, so only spectral *shape* differences (the MTF) remain
  between the two patches (spec §5.3).

:func:`estimate_empty_background` provides the robust dark level and noise
sigma of a mostly-empty tile (noise floor for the dz fit, spec §5.3).
"""

from __future__ import annotations

import numpy as np

__all__ = ["l1_normalize", "l2_match", "estimate_empty_background"]

_L2_DEFAULT_F_MAX = 0.15
_MAD_TO_SIGMA = 1.4826  # 1/Phi^-1(0.75): median absolute deviation -> sigma


def l1_normalize(
    tile: np.ndarray, kind: str
) -> tuple[np.ndarray, float, float]:
    """Normalize one tile to zero background and unit signal scale (L1).

    Fluorescence (dark background): ``bg = median(tile[tile <= median(tile)])``,
    ``scale = p99.9 - bg``, return ``(tile - bg) / scale``.
    Transmission (bright background): ``bg = p50`` (the well bottom),
    ``scale = bg - p1``, return ``(bg - tile) / scale`` — i.e. darkness, so
    tissue reads bright exactly as in the fluorescence channels.

    Returns ``(normalized, bg, scale)`` with the image as float32 and empty
    background at 0. Raises :class:`ValueError` for an unknown ``kind`` or
    when ``scale <= 0`` (no usable signal range in the tile).
    """
    tile = np.asarray(tile)
    if kind == "fluorescence":
        bg = float(np.median(tile[tile <= np.median(tile)]))
        scale = float(np.percentile(tile, 99.9)) - bg
    elif kind == "transmission":
        bg = float(np.percentile(tile, 50))
        scale = bg - float(np.percentile(tile, 1))
    else:
        raise ValueError(f"unknown channel kind: {kind!r}")
    if scale <= 0:
        raise ValueError(
            f"l1_normalize: degenerate scale {scale} for kind {kind!r}; "
            "tile has no usable signal range"
        )
    if kind == "fluorescence":
        out = (tile - bg) / scale
    else:
        out = (bg - tile) / scale
    return out.astype(np.float32), bg, scale


def l2_match(
    ref: np.ndarray, target: np.ndarray, f_max: float | None = None
) -> tuple[np.ndarray, float, float]:
    """Match ``target`` to ``ref`` with a low-frequency gain/offset fit (L2).

    A low-pass mask (rfft2 radius <= ``f_max`` cycles/px, default 0.15)
    selects the frequencies below the noise knee; on the masked inverse FFTs
    of both patches the model ``target ≈ g·ref + c`` is solved by least
    squares (``np.linalg.lstsq`` on ``[masked_ifft(ref), ones]``). The
    *inverse* mapping is then applied to the target (spec §5.3: "apply the
    inverse to the target before comparison"), so the returned patch is a
    copy of ``target`` brought onto the reference's gain/offset and only
    spectral shape differences remain.

    Returns ``(matched, g, c)``. Raises :class:`ValueError` on shape mismatch
    or a degenerate fit (reference with no usable low-frequency structure).
    """
    if f_max is None:
        f_max = _L2_DEFAULT_F_MAX
    ref = np.asarray(ref, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if ref.shape != target.shape:
        raise ValueError(
            f"l2_match: shape mismatch ref {ref.shape} vs target {target.shape}"
        )
    fy = np.fft.fftfreq(ref.shape[0])[:, None]
    fx = np.fft.rfftfreq(ref.shape[1])[None, :]
    mask = np.hypot(fy, fx) <= f_max
    ref_low = np.fft.irfft2(np.fft.rfft2(ref) * mask, s=ref.shape)
    target_low = np.fft.irfft2(np.fft.rfft2(target) * mask, s=target.shape)
    design = np.column_stack([ref_low.ravel(), np.ones(ref_low.size)])
    fit, _, rank, _ = np.linalg.lstsq(design, target_low.ravel(), rcond=None)
    g, c = float(fit[0]), float(fit[1])
    if rank < 2 or not np.isfinite(g) or g == 0:
        raise ValueError(
            "l2_match: degenerate low-frequency fit; the reference patch "
            "has no usable structure to match against"
        )
    matched = ((target - c) / g).astype(np.float32)
    return matched, g, c


def estimate_empty_background(tile: np.ndarray) -> tuple[float, float]:
    """Estimate the background level and noise sigma of a mostly-empty tile.

    ``med`` is the tile median; ``sig`` is the MAD-based sigma
    (``1.4826 · median(|tile - med|)``) restricted to the lowest 50 %
    intensities, so small bright objects cannot inflate the noise estimate.

    Returns ``(med, sig)`` as plain floats. ``sig`` is 0 for a perfectly
    flat background (no noise in the data itself).
    """
    tile = np.asarray(tile, dtype=np.float64)
    med = float(np.median(tile))
    low = tile[tile <= np.percentile(tile, 50)]
    sig = float(_MAD_TO_SIGMA * np.median(np.abs(low - med)))
    return med, sig
