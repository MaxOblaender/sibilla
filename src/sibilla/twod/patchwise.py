"""Patchwise defocus recovery: patch grid, per-patch pipeline, feathered blend (spec §7).

A PCLS tile has *spatially varying* defocus, so it cannot be deconvolved
with one PSF: the tile is cut into overlapping patches, each patch gets its
own ``dz`` (estimated against the well-level in-focus reference) and its own
Richardson–Lucy deconvolution, and the patch results are re-assembled with a
feathered (raised-cosine) window so patch boundaries do not ring.

Per patch, :func:`deconvolve_patchwise` runs the spec §5–§7 chain::

    L1 (once, whole tile) → L2 match against ref → estimate dz
    → RL with the patch's own post-L2 dark level as background
    → accumulate with the feather window

Load-bearing rules:

- **L2 before the spectral comparison and before RL** (spec §5.3): the
  gain/offset fit is applied per (target, reference) pair, so only spectral
  *shape* differences (the MTF) remain between patch and reference.
- **RL background = the patch's own dark level after L2** (spec §7): the
  pedestal must sit in the frame RL is handed (the L2-matched patch), or
  the constant is at the wrong level. Clamped at 0 — RL requires a
  non-negative additive pedestal.
- **Unrecoverable patches are skipped, not deconvolved** (spec §5.5): a
  collapsed recovery horizon means RL would only amplify noise; the patch
  passes through at its L2-matched value and the record says so
  (``unrecoverable=True``, ``iterations=0``, ``recovery=None``).
- **Indeterminate dz patches are skipped, not deconvolved** (round-4
  ruling): with ``min_dz_identification > 0`` a patch whose dz fit cannot
  exclude dz = 0 at that error level (``indeterminate=True``,
  ``dz_um=nan``) carries no identifiable defocus signal in its usable
  band — RL with an assumed PSF would be as wrong as RL with no PSF — so
  it takes the same skip-RL path as an unrecoverable patch, with its own
  record class (``indeterminate=True``, ``unrecoverable=False``). The
  default ``min_dz_identification=0.0`` keeps the historical behaviour
  (every fittable patch is deconvolved with whatever dz the fit returns).
- **``run_rl=False`` is the background-only control arm** (spec §2
  three-arm comparison): the identical code path — L1, per-patch L2, same
  feathered reassembly — with dz estimation and RL skipped; records carry
  ``dz_um=nan``, ``iterations=0``, ``recovery=None``. The control differs
  from the full arm ONLY by the deconvolution.
- **RL early-stops on the §7 recovery metric**: the metric handed to RL is
  the NEGATIVE band-limited spectral distance of the running estimate to
  the reference patch, over the dz fit band's bins (the reference's band
  spectrum is computed once per tile), tracked best-so-far so rl2d's
  consecutive-stall detector sees oscillating divergence too (measured
  second divergence class; see :func:`_early_stop_metric`). Higher =
  better: convergence shrinks the distance so the metric rises, then
  plateaus; divergence grows the distance so the metric drops. With
  ``tol=0.02``/``patience=5`` (the §7 values) both pathologies stop early —
  without a metric, divergence runs all ``n_iter`` iterations and explodes
  (measured on W042: 1e55 recovery spikes).

``DzResult.haze_dominated`` is reported but not gated: pilot policy is to
record it in run_meta, not act on it.

Noise for the dz fit: ``noise_sigma`` is measured once from the
L1-normalized tile with :func:`sibilla.twod.flatfield.estimate_empty_background`
(spec §5.4: "noise estimated from empty background regions of the same
tile") and rescaled into each matched patch's units by ``1/|g|`` (the L2
inverse is affine, so it rescales noise exactly like signal; the ratio and
knee tests inside :func:`sibilla.twod.dz_estimate.estimate_dz` are
scale-invariant only if sigma is rescaled with it). On an 8-bit tile whose
background is *exactly* flat the MAD sigma is 0; a 1.0 DN noise floor
(expressed in L1 units) then substitutes as a *conditional* floor — σ = 0
would make the knee test pass at every frequency and let RL amplify noise
without bound (measured on W042: 1e12–1e55 recovery spikes). The floor is
1.0 DN — the 8-bit quantization noise ``ΔDN/√12 = 0.289`` DN plus
realistic read noise — because quantization alone proved *functionally
inert* here (pilot round 2: 4σ² ≈ 5e-6, near-flat patches with faint haze
still passed the knee and RL still diverged). It remains deliberately
conditional rather than an unconditional ``max()``: at ``1.0/scale`` it is
hundreds× the true σ of any continuous synthetic fixture and would
collapse every recovery horizon (see the comment at the wiring site). The estimator
assumes the tile is *mostly empty* (true for fluorescence PCLS tiles); on
dense textures the MAD sigma measures texture spread instead of noise,
which pushes patches toward ``unrecoverable`` — the conservative direction.

Centring (Task 5 contract): :func:`sibilla.twod.psf2d.psf2d` returns the
kernel with its peak at ``[G//2, G//2]`` and :func:`sibilla.twod.rl2d.rl2d`
uses the conjugate-FFT adjoint consistent with that roll, so blur and
deconvolution share one frame. Full-size PSFs are used here
(``grid_size=size``, no cropping); centre-cropping the un-rolled PSF as an
optimization would shift images by half a patch.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import dz_estimate, flatfield, rl2d
from .optics import CHANNELS
from .psf2d import psf2d

__all__ = [
    "PatchRecord",
    "patch_grid",
    "feather_weights",
    "recovery_metric",
    "deconvolve_patchwise",
]

#: floor on the accumulated weight map (defensive; the map is >= 0 everywhere)
_WSUM_EPS = 1e-12


@dataclass
class PatchRecord:
    """Per-patch QC record (spec §9), one per patch of the grid.

    row, col : grid position of the patch.
    y0, x0, size : top-left corner and side length in tile pixels.
    dz_um, dz_err_um : defocus estimate and 1σ error (µm). NaN when the dz
        fit was rejected and 0.0-with-``unrecoverable=True`` is recorded
        instead (spec: "dz=0 + unrecoverable=True"); NaN for the control
        arm, where dz estimation is skipped entirely.
    f_horizon : recovery horizon from the dz fit (cyc/px); NaN when not
        computed (control arm).
    iterations : RL iterations actually applied (0 when RL was skipped).
    g, c : the L2 gain/offset fit of this patch against the reference.
    recovery : high-frequency recovery metric after/before RL; ``None``
        whenever RL did not run (control arm or unrecoverable patch).
    unrecoverable : the estimator's horizon flag — the patch was passed
        through without deconvolution.
    indeterminate : the estimator's "no opinion" flag (round 4; last
        field): the dz fit could not exclude dz = 0 at
        ``min_dz_identification`` — the patch is informationless for
        deconvolution and was passed through without RL
        (``dz_um=nan``, ``iterations=0``, ``recovery=None``,
        ``unrecoverable=False``). Always False for the control arm (dz
        not attempted).
    """

    row: int
    col: int
    y0: int
    x0: int
    size: int
    dz_um: float
    dz_err_um: float
    f_horizon: float
    iterations: int
    g: float
    c: float
    recovery: float | None
    unrecoverable: bool
    indeterminate: bool = False


def _starts(n: int, size: int, step: int) -> list[int]:
    """Patch start coordinates along one axis.

    Starts run from 0 on the ``step`` lattice; when the lattice would end
    short of the edge, one final start is clamped to ``n - size`` so the
    last patch ends exactly at the image edge. An axis shorter than one
    patch gets the single start 0 (the caller pads to ``size`` — even-sized
    patches are required by the full-size-PSF centring contract).
    """
    if n <= size:
        return [0]
    starts = list(range(0, n - size + 1, step))
    if starts[-1] != n - size:
        starts.append(n - size)
    return starts


def patch_grid(shape: tuple[int, int], size: int = 512,
               overlap: float = 0.25) -> list[tuple[int, int, int]]:
    """Patch lattice covering ``shape``: ``(index, y0, x0)`` per patch.

    Step is ``round(size * (1 - overlap))`` px (25 % overlap → 384 px for
    512-px patches). Starts begin at 0 on every axis and the last patch on
    each axis is clamped so it ends exactly at the image edge; the union of
    patches covers the whole image and every patch is exactly ``size`` px.
    ``index`` enumerates patches in row-major order (``row = index //
    n_cols``, ``col = index % n_cols``).
    """
    if size <= 0:
        raise ValueError(f"patch_grid: size must be > 0, got {size}")
    if not 0.0 <= overlap < 1.0:
        raise ValueError(f"patch_grid: overlap must be in [0, 1), got {overlap}")
    ny, nx = int(shape[0]), int(shape[1])
    step = max(1, int(round(size * (1.0 - overlap))))
    ys = _starts(ny, size, step)
    xs = _starts(nx, size, step)
    return [(iy * len(xs) + ix, y, x)
            for iy, y in enumerate(ys) for ix, x in enumerate(xs)]


def feather_weights(size: int, overlap: float) -> np.ndarray:
    """Separable raised-cosine feather window, ``(size, size)`` in ``[0, 1]``.

    Each axis is a 1-D window: a raised-cosine ramp over
    ``round(overlap * size)`` px on each side, plateau at 1 inside; the 2-D
    window is the outer product. The ramp is sampled at pixel centres
    (half-sample offset), so the outermost pixel keeps a small positive
    weight instead of exactly 0 — after normalization by the weight sum, a
    pixel covered by a single patch keeps that patch's own value rather
    than being zeroed.
    """
    size = int(size)
    if size <= 0:
        raise ValueError(f"feather_weights: size must be > 0, got {size}")
    if not 0.0 <= overlap < 1.0:
        raise ValueError(
            f"feather_weights: overlap must be in [0, 1), got {overlap}"
        )
    ramp = min(int(round(overlap * size)), size // 2)
    w = np.ones(size, dtype=np.float64)
    if ramp > 0:
        t = (np.arange(ramp) + 0.5) / ramp
        up = 0.5 * (1.0 - np.cos(np.pi * t))
        w[:ramp] = up
        w[size - ramp:] = up[::-1]
    return np.outer(w, w)


def _band_power(img: np.ndarray, f_lo: float, f_hi: float) -> float:
    """Sum of azimuthally averaged power in ``[f_lo, f_hi]`` cyc/px."""
    img = np.asarray(img, dtype=np.float64)
    bins = dz_estimate.radial_bins(img.shape)
    freqs, power = dz_estimate.radial_spectrum(img, bins)
    band = (freqs >= f_lo) & (freqs <= f_hi) & np.isfinite(power)
    if not band.any():
        return 0.0
    return float(np.sum(power[band]))


def recovery_metric(before: np.ndarray, after: np.ndarray,
                    f_lo: float = 0.025, f_hi: float = 0.5) -> float:
    """High-frequency recovery: band power after / before (spec §9).

    Both arrays are radially binned (:func:`dz_estimate.radial_bins`) and
    the summed per-mode power over ``[f_lo, f_hi]`` cyc/px is compared:
    ~1.0 means the band is unchanged, > 1 that energy was restored in it
    (deblurring a defocused patch raises high-frequency power). This
    implements the plain after/before ratio of spec §9; the spec's
    normalization by the in-focus reference's band energy and the
    ``f_hi = min(f_horizon, f_max)`` band cap are deliberately not
    implemented (plan-sanctioned simplification). Returns ``nan`` when the
    band is empty or the "before" power is ~0 (ratio undefined — a flat
    patch has no band power to recover).
    """
    p_before = _band_power(before, f_lo, f_hi)
    p_after = _band_power(after, f_lo, f_hi)
    if p_before <= 0.0:
        return float("nan")
    return p_after / p_before


def _band_distance(x: np.ndarray, bins, band: np.ndarray,
                   ref_band: np.ndarray) -> float:
    """NEGATIVE band-limited spectral distance to the reference (spec §7).

    Metric value for :func:`_early_stop_metric`:
    :func:`dz_estimate.radial_spectrum` of the running estimate compared to
    the reference patch's precomputed band spectrum over the dz fit band's
    bins. RL converging toward the reference statistics shrinks the
    distance (value rises toward 0, then plateaus); RL diverging grows it
    (value drops). Returns ``-inf`` for a non-finite (overflowing)
    distance so the plateau comparison still reads "no improvement".
    """
    _, p = dz_estimate.radial_spectrum(x, bins)
    d = p[band] - ref_band
    d = d[np.isfinite(d)]
    if d.size == 0:
        return 0.0
    with np.errstate(over="ignore"):
        dist = float(d @ d)
    return -dist if np.isfinite(dist) else float("-inf")


def _early_stop_metric(bins, band: np.ndarray, ref_band: np.ndarray):
    """§7 early-stop metric for ONE RL run (factory; one call per patch).

    Returns ``metric(x) -> float`` for :func:`rl2d.rl2d` (higher is
    better): the NEGATIVE band-limited spectral distance to the reference
    patch (:func:`_band_distance`), tracked as a best-so-far ENVELOPE.

    The envelope is a MEASURED deviation from handing rl2d the raw metric
    value, flagged in the Task-9 round-3 report: RL divergence comes in two
    classes on the tissue fixture — monotone (distance grows every
    iteration; the raw metric stops at iteration 6) and OSCILLATING
    (distance swings e.g. −841 → −533 → −649 → −418: every second
    iteration "improves" by ≫ 2 % and resets rl2d's consecutive-stall
    counter, so the raw value runs all 30 iterations on a bounded
    oscillation). Tracking the best-so-far makes §7's "stops improving by
    > 2 % over 5 iterations" mean "no new best for 5 iterations", which
    stops both classes while leaving genuinely converging runs untouched —
    the distance shrinks monotonically there, so envelope ≡ raw value
    (measured: identical stop iteration on the convergent fixture patch).
    """
    best = float("-inf")

    def metric(x: np.ndarray) -> float:
        nonlocal best
        m = _band_distance(x, bins, band, ref_band)
        if m > best:
            best = m
        return best

    return metric


def _dark_level(matched: np.ndarray) -> float:
    """RL background: the patch's own dark level after L2 (spec §7).

    ``estimate_empty_background``'s median of the L2-matched patch — after
    L1 + L2 an empty region sits at ~0, so this is the residual pedestal in
    exactly the frame RL sees. Clamped at 0 (RL requires ``background >=
    0``; a negative median means the patch is already zero-level).
    """
    med, _ = flatfield.estimate_empty_background(matched)
    return max(float(med), 0.0)


def _sharpest_patch(norm: np.ndarray, grid, size: int) -> np.ndarray:
    """Sharpest patch of the tile (the in-tile reference fallback)."""
    best_score, best = -np.inf, None
    for _, y0, x0 in grid:
        cand = norm[y0:y0 + size, x0:x0 + size]
        score = dz_estimate.sharpness(cand)
        if score > best_score:
            best_score, best = score, cand
    return np.asarray(best, dtype=np.float32)


def deconvolve_patchwise(tile: np.ndarray, channel: str,
                         ref_patch: np.ndarray | None = None,
                         coeffs: dict[int, float] | None = None,
                         size: int = 512, overlap: float = 0.25,
                         n_iter: int = 30,
                         run_rl: bool = True,
                         min_dz_identification: float = 0.0,
                         ) -> tuple[np.ndarray, list[PatchRecord]]:
    """Deconvolve one tile patch by patch; reassemble with feathering.

    Parameters
    ----------
    tile : 2D tile (any numeric dtype); processed in float32 on
        L1-normalized data (spec §7), returned in the same normalized units
        (empty background ≈ 0, unit signal scale).
    channel : Keyence channel name (``sibilla.twod.optics.CHANNELS``); its
        ``kind`` selects the L1 normalization sign (fluorescence vs
        transmission-imported darkness).
    ref_patch : well-level in-focus reference (Task 9), used as supplied —
        the per-patch L2 match brings every patch onto the reference's
        gain/offset, so its units (raw DN or L1-normalized) do not matter,
        and ``noise_sigma`` is rescaled by ``1/|g|`` into the matched
        frame. ``None`` uses the sharpest patch of the tile itself
        (single-tile/unit-test use); the records' ``g``/``c`` then reflect
        that in-tile choice.
    coeffs : ``{Noll index: RMS waves}`` passed through to :func:`psf2d`
        (Zernikes = 0 in the pilot, i.e. ``None``).
    size, overlap : patch side length and fractional overlap (spec §7:
        512 px, 25 %). ``overlap * size`` is the feather ramp per side.
    n_iter : maximum RL iterations per recoverable patch; the §7 early
        stop (negative band-limited spectral distance to the reference
        patch, ``tol=0.02``, ``patience=5``) usually ends RL sooner, and
        :attr:`PatchRecord.iterations` reports the iterations actually
        applied.
    run_rl : ``False`` = background-only control arm (see module docstring).
    min_dz_identification : µm; passed through to
        :func:`sibilla.twod.dz_estimate.estimate_dz`. 0.0 (default) keeps
        the historical behaviour (every fittable patch is deconvolved with
        whatever dz the fit returns). With ``m > 0``, a patch whose dz fit
        cannot exclude dz = 0 (``dz_err_um >= m`` and ``|dz| < dz_err_um``,
        or a non-finite error bar from a singular Jacobian) is
        *indeterminate* — informationless for deconvolution — and takes
        the same skip-RL path as an unrecoverable patch: ``dz_um=nan``,
        ``iterations=0``, ``recovery=None``, record
        ``indeterminate=True`` (never ``unrecoverable=True``; the two
        classes stay distinct). The control arm never sets it (dz is not
        attempted there).

    Returns ``(out, records)``: the reassembled tile (float32, tile shape)
    and one :class:`PatchRecord` per patch of ``patch_grid(tile.shape,
    size, overlap)``.

    Tiles smaller than ``size`` on an axis are symmetric-padded up to
    ``size`` (content anchored at the origin, mirrored fill beyond), so
    every patch handed to L2/RL is exactly ``size × size``; the result is
    cropped back to the tile.

    Raises :class:`ValueError` for a non-2D tile, a ``ref_patch`` whose
    shape is not ``(size, size)``, an invalid ``size``/``overlap``, or a
    degenerate L1 normalization (flat tile); :class:`KeyError` for an
    unknown channel.
    """
    if channel not in CHANNELS:
        raise KeyError(
            f"unknown channel {channel!r}; known: {sorted(CHANNELS)}"
        )
    kind = CHANNELS[channel].kind
    size = int(size)
    if size <= 0:
        raise ValueError(f"deconvolve_patchwise: size must be > 0, got {size}")

    raw = np.asarray(tile)
    if raw.ndim != 2:
        raise ValueError(
            f"deconvolve_patchwise: expected a 2D tile, got {raw.shape}"
        )
    ny, nx = raw.shape

    # --- L1 once for the whole tile: background -> 0, comparable contrast ---
    norm, _bg, scale = flatfield.l1_normalize(raw, kind)

    # noise floor for the dz fit, from the tile's empty background (§5.4);
    # rescaled per patch by 1/|g| into the L2-matched frame.
    #
    # Degenerate-sigma guard — MEASURED DECISION (conditional, not an
    # unconditional max()): on the synthetic fixtures the true normalized
    # background sigma is ~0.005 while 1.0/scale ~ 2 — an unconditional
    # max() would make the floor dominate by ~400x, collapse the recovery
    # horizon (4*sigma^2 ~ 16 at every frequency) and render the
    # background-only and full arms identical (pinned arm test fails,
    # measured). So the floor substitutes ONLY when the MAD sigma is
    # exactly 0 — the production pathology this guards: an 8-bit tile whose
    # background is exactly flat (every background pixel rounds to the same
    # DN; W042 P00001/P00007: 6 DN everywhere) gives sigma = 0 -> the knee
    # test passes everywhere -> unguarded RL amplifies noise into
    # 1e12-1e55 "recovery" spikes. The floor is 1.0 DN — quantization
    # (DeltaDN/sqrt(12) = 0.289 DN) plus realistic read noise — expressed
    # here in L1 units (/ scale). The pure-quantization value 0.289 DN was
    # measured functionally inert in pilot round 2 (4*sigma^2 ~ 5e-6:
    # near-flat patches with faint haze still passed the knee and RL still
    # diverged); 1.0 DN covers both contributions conservatively.
    _, sigma_tile = flatfield.estimate_empty_background(norm)
    if sigma_tile <= 0.0:
        sigma_tile = 1.0 / scale

    # tiles smaller than one patch: symmetric-pad so patches stay size x size
    pad_y, pad_x = max(0, size - ny), max(0, size - nx)
    work = np.pad(norm, ((0, pad_y), (0, pad_x)), mode="symmetric") \
        if (pad_y or pad_x) else norm
    ph, pw = work.shape

    grid = patch_grid((ph, pw), size, overlap)

    # --- reference patch: well-level (caller) or the tile's sharpest ---
    if ref_patch is None:
        ref = _sharpest_patch(work, grid, size)
    else:
        ref = np.asarray(ref_patch)
        if ref.shape != (size, size):
            raise ValueError(
                f"deconvolve_patchwise: ref_patch must be {size}x{size} "
                f"(the patch size), got {ref.shape}"
            )
        ref = ref.astype(np.float32)

    # object-spectrum anchor + bin-centre frequencies from the reference
    # (== build_object_spectrum([ref], bins): the median of one patch)
    bins = dz_estimate.radial_bins((size, size))
    freqs, s_obj = dz_estimate.radial_spectrum(ref, bins)

    # §7 early-stop metric support: the reference's band-limited radial
    # spectrum is computed ONCE per tile (s_obj above); the low band edge
    # (the dz fit's F_LO) is per tile too — per patch only the upper edge
    # (the dz fit's own f_horizon) masks the band.
    finite_band = np.isfinite(s_obj) & (freqs >= dz_estimate.F_LO)

    w2 = feather_weights(size, overlap).astype(np.float32)
    acc = np.zeros((ph, pw), dtype=np.float64)
    wsum = np.zeros((ph, pw), dtype=np.float64)
    records: list[PatchRecord] = []
    n_cols = len({x for _, _, x in grid})

    for idx, y0, x0 in grid:
        row, col = divmod(idx, n_cols)
        patch = np.asarray(work[y0:y0 + size, x0:x0 + size], dtype=np.float32)

        # L2: bring the patch onto the reference's gain/offset (§5.3)
        matched, g, c = flatfield.l2_match(ref, patch)
        matched = np.asarray(matched, dtype=np.float32)

        if not run_rl:  # background-only control arm
            records.append(PatchRecord(
                row=row, col=col, y0=y0, x0=x0, size=size,
                dz_um=float("nan"), dz_err_um=float("nan"),
                f_horizon=float("nan"), iterations=0, g=g, c=c,
                recovery=None, unrecoverable=False, indeterminate=False,
            ))
            acc[y0:y0 + size, x0:x0 + size] += w2 * matched
            wsum[y0:y0 + size, x0:x0 + size] += w2
            continue

        # l2_match passes degenerate fits (|g| < GAIN_FLOOR) through uncorrected;
        # the patch then keeps its own frame, so no 1/|g| noise rescale.
        sigma = sigma_tile / abs(g) if abs(g) >= flatfield.GAIN_FLOOR else sigma_tile
        dz = dz_estimate.estimate_dz(matched, ref, s_obj, freqs, channel, sigma,
                                     min_dz_identification=min_dz_identification)

        if dz.unrecoverable:  # skip RL: pass the matched patch through
            records.append(PatchRecord(
                row=row, col=col, y0=y0, x0=x0, size=size,
                dz_um=0.0, dz_err_um=float("nan"),
                f_horizon=dz.f_horizon, iterations=0, g=g, c=c,
                recovery=None, unrecoverable=True,
            ))
            acc[y0:y0 + size, x0:x0 + size] += w2 * matched
            wsum[y0:y0 + size, x0:x0 + size] += w2
            continue

        if dz.indeterminate:
            # "no opinion" (round-4 ruling): the fit cannot exclude dz = 0,
            # so the patch carries no identifiable defocus signal — RL with
            # an assumed PSF would be as wrong as RL with no PSF. Same
            # skip-RL path as unrecoverable, but its own record class (the
            # dz fit DID run: f_horizon/dz_err are meaningful QC, dz is not).
            records.append(PatchRecord(
                row=row, col=col, y0=y0, x0=x0, size=size,
                dz_um=float("nan"), dz_err_um=dz.dz_err_um,
                f_horizon=dz.f_horizon, iterations=0, g=g, c=c,
                recovery=None, unrecoverable=False, indeterminate=True,
            ))
            acc[y0:y0 + size, x0:x0 + size] += w2 * matched
            wsum[y0:y0 + size, x0:x0 + size] += w2
            continue

        psf = psf2d(dz.dz_um, channel, grid_size=size, coeffs=coeffs)
        # §7 early stop ("recovery metric stops improving by >2 % over 5
        # iterations"): metric = NEGATIVE band-limited spectral distance to
        # the reference patch over this patch's dz fit band (its reference
        # spectrum was computed once per tile), best-so-far-tracked so the
        # plateau detector sees both divergence classes (see
        # _early_stop_metric).
        band = finite_band & (freqs <= dz.f_horizon)
        res = rl2d.rl2d(matched, psf, background=_dark_level(matched),
                        n_iter=n_iter,
                        metric=_early_stop_metric(bins, band, s_obj[band]),
                        tol=0.02, patience=5)
        after = np.asarray(res.image, dtype=np.float32)
        records.append(PatchRecord(
            row=row, col=col, y0=y0, x0=x0, size=size,
            dz_um=dz.dz_um, dz_err_um=dz.dz_err_um, f_horizon=dz.f_horizon,
            iterations=int(res.iterations), g=g, c=c,
            recovery=recovery_metric(matched, after), unrecoverable=False,
        ))
        acc[y0:y0 + size, x0:x0 + size] += w2 * after
        wsum[y0:y0 + size, x0:x0 + size] += w2

    out = (acc / np.maximum(wsum, _WSUM_EPS))[:ny, :nx]
    return out.astype(np.float32), records
