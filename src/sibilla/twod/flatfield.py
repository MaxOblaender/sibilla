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
sigma of a mostly-empty tile (noise floor for the dz fit, spec §5.3), both
measured on the dimmest quartile, with an optional caller-supplied sigma
floor (8-bit quantization noise).
"""

from __future__ import annotations

import numpy as np

__all__ = ["l1_normalize", "l2_match", "estimate_empty_background", "GAIN_FLOOR"]

_L2_DEFAULT_F_MAX = 0.15
# MAD -> sigma for the *bottom-quartile* (truncated Gaussian) sample that
# estimate_empty_background measures: MAD of a unit normal restricted to its
# lowest 25 % is 0.3026 (Monte Carlo, 4e6 draws), so sigma = MAD / 0.3026.
# (The familiar 1.4826 is the FULL-Gaussian factor; applied to the quartile
# it understates sigma by 2.3x — measured: the spots fixture's 6-sigma
# detection threshold collapsed to 2.6 sigma true, 3600 noise pixels seeded,
# pinned spot detection died by region merging.)
_QUARTILE_MAD_TO_SIGMA = 3.3043
GAIN_FLOOR = 0.1  # fits with |g| below this are degenerate: never apply the inverse


def l1_normalize(
    tile: np.ndarray, kind: str
) -> tuple[np.ndarray, float, float]:
    """Normalize one tile to zero background and unit signal scale (L1).

    Fluorescence (dark background): ``bg = median(tile[tile <= median(tile)])``,
    ``scale = p99.9 - bg``, return ``(tile - bg) / scale``.
    Transmission (bright background): ``bg = p90`` — the illuminated well
    bottom, ``scale = bg - p1``, return ``(bg - tile) / scale`` — i.e.
    darkness, so tissue reads bright exactly as in the fluorescence channels.

    Why p90 and not p50: ``p50`` assumed the well bottom occupies at least
    half the tile. MEASURED on the W042 CH4 pilot tiles that assumption
    fails — P00001/P00007: ``p50 = 1`` DN with ~2/3 of the tile near-black
    (the illuminated well bottom is the bright minority, p75–p99 = 34–150
    DN), so ``scale = p50 - p1`` collapsed to 1–4 DN and the background
    texture mapped to σ ≈ 40–76 normalized units (every patch unrecoverable;
    pilot round 2). ``p90`` sits inside the bright mode in both regimes —
    measured 59–146 DN on all nine W042 CH4 tiles (vs p50 = 1–107), and
    exactly the old value on well-bottom-dominated tiles (pinned fixture:
    bg 240, dark square → p50 = p90 = 240).

    Returns ``(normalized, bg, scale)`` with the image as float32 and empty
    background at 0. Raises :class:`ValueError` for an unknown ``kind`` or
    when ``scale <= 0`` (no usable signal range in the tile).
    """
    tile = np.asarray(tile)
    if kind == "fluorescence":
        bg = float(np.median(tile[tile <= np.median(tile)]))
        scale = float(np.percentile(tile, 99.9)) - bg
    elif kind == "transmission":
        bg = float(np.percentile(tile, 90))
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
    selects the frequencies below the noise knee; ``f_max`` >= Nyquist
    (0.5 cyc/px) degenerates to a full-band fit — no knee left to respect.
    On the masked inverse FFTs
    of both patches the model ``target ≈ g·ref + c`` is solved by least
    squares (``np.linalg.lstsq`` on ``[masked_ifft(ref), ones]``). The
    *inverse* mapping is then applied to the target (spec §5.3: "apply the
    inverse to the target before comparison"), so the returned patch is a
    copy of ``target`` brought onto the reference's gain/offset and only
    spectral shape differences remain.

    Returns ``(matched, g, c)``. The correction is applied only when the fit
    is non-degenerate: if the fitted gain is non-finite or ``|g| < GAIN_FLOOR``
    (e.g. an empty patch vs a textured reference), the inverse
    ``(target - c)/g`` would explode the patch by ``1/|g|``, so ``matched``
    is an uncorrected copy of ``target`` while ``g``/``c`` still report the
    fitted values for the record. Raises :class:`ValueError` on shape
    mismatch or a rank-deficient fit (reference with no usable
    low-frequency structure).
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
    if rank < 2:
        raise ValueError(
            "l2_match: degenerate low-frequency fit; the reference patch "
            "has no usable structure to match against"
        )
    if not np.isfinite(g) or abs(g) < GAIN_FLOOR:
        # Degenerate gain (e.g. empty patch vs textured reference): the
        # inverse (target - c)/g would explode the patch by 1/|g|. Pass the
        # target through UNCORRECTED; g/c still report the fitted values.
        return target.copy(), g, c
    matched = ((target - c) / g).astype(np.float32)
    return matched, g, c


def estimate_empty_background(
    tile: np.ndarray, *, sigma_floor: float | None = None
) -> tuple[float, float]:
    """Estimate the background level and noise sigma of a mostly-empty tile.

    Both statistics are measured on the **lowest-25 % intensity subset** — the
    dimmest quartile is the empty-background population:

    - ``med`` is that subset's own median: the **background level**. It is
      *not* the full-tile median: on a tile whose overall median sits far
      above the dark level (a bright or hazy lower half), centring on the
      full-tile median folds the object/background offset into ``sig``
      (measured on W042: 30–172 DN-equivalent vs a true 1.5–4.5 DN).
    - ``sig`` is the MAD-based noise sigma of that subset around ``med``
      (MAD recalibrated for the truncated quartile: ``MAD / 0.3026``, see
      ``_QUARTILE_MAD_TO_SIGMA``): the **robust noise std** of the
      empty background; bright objects cannot inflate it.

    ``sigma_floor`` (keyword-only, ``None`` = no floor): when given,
    ``sig = max(sig, sigma_floor)``. Callers that know the quantization /
    detector noise floor (e.g. ``ΔDN / sqrt(12) / scale``) use it to keep
    ``sig`` from collapsing to its degenerate MAD == 0 value on a perfectly
    flat 8-bit background.

    Returns ``(med, sig)`` as plain floats. Without a floor, ``sig`` is 0 for
    a perfectly flat background (no noise in the data itself).
    """
    tile = np.asarray(tile, dtype=np.float64)
    low = tile[tile <= np.percentile(tile, 25)]
    med = float(np.median(low))
    sig = float(_QUARTILE_MAD_TO_SIGMA * np.median(np.abs(low - med)))
    if sigma_floor is not None:
        sig = max(sig, float(sigma_floor))
    return med, sig
