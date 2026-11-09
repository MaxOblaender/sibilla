"""In-focus-anchored spectral defocus estimator (spec §5, the 2D core).

Physics: a patch blurred by defocus ``dz`` has power spectrum
``S_obs(f) = S_obj(f) · |MTF(f; dz)|²`` with the *same* object spectrum as
the in-focus region of the same well. The object spectrum is therefore
anchored empirically from in-focus reference patches and ``dz`` follows by
fitting the spectral ratio ``R(f) = S_obs/S_obj`` against the model MTF²
family from :func:`sibilla.twod.psf2d.psf2d`. A free power-law ``slope``
(``log R = log MTF²(dz) + slope·log f``) absorbs object-statistics
mismatch between the patches, and the known noise power ``noise_sigma²``
is subtracted from the observed spectrum before the ratio (an additive
noise floor would otherwise flatten the apparent decay and bias dz low).
The fit runs on the band ``[F_LO, f_knee]`` where ``f_knee`` is the
highest frequency whose observed power still exceeds ``KNEE_SNR`` times
the noise power — above the knee the ratio measures noise, not defocus.

Two QC flags protect the downstream Richardson–Lucy step:

- ``f_horizon`` is the highest frequency where ``R`` stays above the noise
  floor ``noise_sigma²/S_obj`` (the recovery horizon). Patches whose
  horizon lies below ``F_LO`` (nothing left to fit) are flagged
  ``unrecoverable``: RL on them would only amplify noise, so the flag is a
  safety feature, not a failure.
- ``haze_dominated`` marks ratios that are not a monotone MTF decay
  (out-of-focus bulk haze adds a low-frequency pedestal), detected via the
  fitted slope/residuals.

Spectrum normalization: :func:`radial_spectrum` returns per-mode power
(normalized by the pixel count), so *white Gaussian noise of std ``σ``
contributes exactly ``σ²`` per mode* — this is the normalization in which
the ``noise_sigma²`` comparisons of the knee/horizon logic live.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import laplace
from scipy.optimize import least_squares

from .psf2d import psf2d

__all__ = [
    "DzResult",
    "radial_bins",
    "radial_spectrum",
    "mtf2_radial",
    "estimate_dz",
    "sharpness",
    "select_references",
    "build_object_spectrum",
]

#: lowest frequency (cyc/px) used in the dz fit — below this the ratio is
#: dominated by offset/gain residuals, not by the MTF (spec §5.3)
F_LO: float = 0.025
#: dz fit upper bound, µm (beyond ~120 µm the MTF is flat zero in-band)
DZ_MAX: float = 120.0
#: minimum number of usable bins in the fit band
MIN_BAND_BINS: int = 5
#: the noise knee is where residual signal falls to this multiple of the
#: noise-floor power; white noise at the floor fluctuates ~±15 % per bin,
#: so a plain ``> noise_sigma²`` threshold never triggers and would admit
#: pure noise (and the pad's spectral leakage) into the fit band
KNEE_SNR: float = 4.0
#: |slope| beyond this means the power-law correction dominates the MTF
#: shape — the ratio is no longer a monotone MTF decay (haze signature)
HAZE_MAX_SLOPE: float = 1.0
#: RMS of the log-ratio residuals beyond which the model does not explain
#: the ratio (haze signature)
HAZE_MAX_RMS: float = 0.5
#: model MTF² floor; |MTF|² below this is log-clamped (fitted dz pinned
#: at the bound instead of producing infinities)
_MTF2_FLOOR: float = 1e-12


@dataclass
class DzResult:
    """Result of :func:`estimate_dz` for one patch.

    dz_um, dz_err_um : defocus magnitude (µm) and its 1σ error from the
        Jacobian (``sqrt(diag(inv(JᵀJ)))``); NaN when unrecoverable, inf
        when the covariance is singular (at the dz = 0 bound the model is
        stationary in dz). The sign of dz is unobservable from a single
        patch's power spectrum (|OTF(−dz)| = |OTF(dz)|), hence the
        magnitude. NaN when ``indeterminate`` (see below).
    slope : fitted power-law exponent of the ratio model (NaN when
        unrecoverable); ~0 for a well-anchored pair of patches.
    f_horizon : highest frequency (cyc/px) where the observed ratio still
        exceeds the noise floor; 0.0 when no bin does.
    unrecoverable : True when the recovery horizon is below ``F_LO`` or
        the fit band has fewer than ``MIN_BAND_BINS`` usable bins.
    haze_dominated : True when the ratio is not a monotone decreasing
        MTF-like decay (see module docstring).
    indeterminate : True when ``min_dz_identification > 0`` and the fit
        cannot distinguish its best-fit dz from zero at that error level
        (:func:`estimate_dz`'s "no opinion" path): ``dz_um`` is then NaN
        and the patch is informationless for deconvolution — downstream
        (:func:`sibilla.twod.patchwise.deconvolve_patchwise`) skips RL on
        it exactly as on an ``unrecoverable`` patch. False when the
        criterion is disabled (the default) or dz = 0 is excluded.
    """

    dz_um: float
    dz_err_um: float
    slope: float
    f_horizon: float
    unrecoverable: bool
    haze_dominated: bool
    indeterminate: bool = False


def radial_bins(shape: tuple[int, int], n_bins: int = 24) -> list[np.ndarray]:
    """Boolean masks of ``n_bins`` linear radial-frequency bins of a 2D FFT.

    The masks index the ``fftshift``-ed power spectrum of an array with
    ``shape``: radii are ``np.hypot`` of the shifted ``fftfreq`` grids and
    bin edges are linear from 0 to 0.5 cyc/px (radial Nyquist; the square
    grid's corners beyond 0.5 fall outside every bin). The last bin is
    closed at 0.5. Returns a list of ``n_bins`` boolean arrays.
    """
    ny, nx = shape
    fy = np.fft.fftshift(np.fft.fftfreq(ny))[:, None]
    fx = np.fft.fftshift(np.fft.fftfreq(nx))[None, :]
    r = np.hypot(fy, fx)
    edges = np.linspace(0.0, 0.5, n_bins + 1)
    masks = []
    for i in range(n_bins):
        if i < n_bins - 1:
            masks.append((r >= edges[i]) & (r < edges[i + 1]))
        else:
            masks.append((r >= edges[i]) & (r <= edges[i + 1]))
    return masks


def radial_spectrum(img: np.ndarray, bins) -> tuple[np.ndarray, np.ndarray]:
    """Radially averaged power spectrum of ``img`` over precomputed ``bins``.

    Computes ``|fftshift(fft2(img))|²``, averages it over each bin mask and
    divides by the pixel count, giving per-mode power: white noise of std
    ``σ`` sits at ``σ²`` (see module docstring). ``bins`` must come from
    :func:`radial_bins` for the same shape. Returns ``(freqs, power)`` with
    ``freqs`` the bin-centre frequencies (cyc/px) and ``power`` NaN for
    empty bins.
    """
    img = np.asarray(img, dtype=np.float64)
    if img.ndim != 2:
        raise ValueError(f"radial_spectrum: expected a 2D image, got {img.shape}")
    power2d = np.abs(np.fft.fftshift(np.fft.fft2(img))) ** 2
    n_bins = len(bins)
    edges = np.linspace(0.0, 0.5, n_bins + 1)
    freqs = 0.5 * (edges[:-1] + edges[1:])
    out = np.full(n_bins, np.nan)
    for i, mask in enumerate(bins):
        if mask.any():
            out[i] = power2d[mask].mean() / img.size
    return freqs, out


def mtf2_radial(dz_um: float, channel: str, freqs: np.ndarray,
                grid_size: int = 256) -> np.ndarray:
    """Model |MTF|² of the channel's PSF at defocus ``dz_um``, per ``freqs``.

    ``|fft2(psf2d(dz))|²`` is radially interpolated (``np.interp`` on the
    sorted radius profile) at the requested frequencies in cyc/px. The PSF
    is energy-normalized, so the OTF is 1 at f = 0. Frequencies are clamped
    to the profile's support (0 … radial Nyquist 0.5).
    """
    freqs = np.asarray(freqs, dtype=np.float64)
    p = psf2d(dz_um, channel, grid_size=grid_size)
    m2 = np.fft.fftshift(np.abs(np.fft.fft2(p)) ** 2)
    fy = np.fft.fftshift(np.fft.fftfreq(grid_size))[:, None]
    fx = np.fft.fftshift(np.fft.fftfreq(grid_size))[None, :]
    r = np.hypot(fy, fx).ravel()
    v = m2.ravel()
    keep = r <= 0.5  # corner frequencies beyond radial Nyquist: unused
    r, v = r[keep], v[keep]
    order = np.argsort(r, kind="stable")
    return np.interp(freqs, r[order], v[order])


def _ratio_model_residuals(params, f_band, y, channel, grid_size):
    dz, slope = params
    m2 = np.clip(mtf2_radial(dz, channel, f_band, grid_size=grid_size),
                 _MTF2_FLOOR, None)
    return y - np.log(m2) - slope * np.log(f_band)


def estimate_dz(target: np.ndarray, ref: np.ndarray,
                object_spectrum: np.ndarray, freqs: np.ndarray,
                channel: str, noise_sigma: float, *,
                min_dz_identification: float = 0.0) -> DzResult:
    """Fit the defocus of ``target`` against the in-focus-anchored ratio.

    Parameters
    ----------
    target : blurred patch whose dz is wanted.
    ref : the in-focus reference patch the anchor came from; accepted for
        interface completeness — the object spectrum is taken from
        ``object_spectrum`` (which the caller derives from ``ref`` via
        :func:`build_object_spectrum` or :func:`radial_spectrum`).
    object_spectrum : per-mode power of the reference object spectrum at
        ``freqs`` (from :func:`radial_spectrum`/:func:`build_object_spectrum`).
    freqs : bin-centre frequencies (cyc/px) matching ``object_spectrum``.
    channel : Keyence channel name (``sibilla.twod.optics.CHANNELS``).
    noise_sigma : image noise std of ``target``; the observed spectrum is
        debiased by the known noise power ``noise_sigma²`` before the
        ratio, and the knee/horizon logic cuts where the residual signal
        approaches the noise floor.
    min_dz_identification : µm; 0.0 (default) disables the indeterminacy
        path (historical behaviour: every fittable patch gets a dz
        number). With ``m > 0``, a fit whose error bar cannot exclude the
        dz = 0 hypothesis is *indeterminate* — the estimator's "no
        opinion" answer for patches that carry no identifiable defocus
        signal in their usable band. Criterion (MEASURED FORMULATION,
        round-4 ruling): indeterminate iff
        ``dz_err_um >= m and abs(dz_um) < dz_err_um`` — the dz = 0
        hypothesis lies within 1σ of the best fit AND the error bar
        itself is at least ``m`` µm. A non-finite error bar (inf/nan from
        a singular Jacobian — the MTF² model is stationary in dz at the
        0 bound, so a fit pinned there has no usable curvature) is always
        indeterminate when ``m > 0``. Measured examples (pinned synthetic:
        smooth object, noise σ = 0.002, ref = object): the dz = 0 case
        fits ``dz = 1e-10, dz_err = inf`` → indeterminate under any
        ``m > 0``; a true dz = 10 µm fits ``dz = 10.26 ± 16.72`` →
        indeterminate under ``m = 5`` (zero within 1σ, error ≥ m) — the
        honest answer, the fit cannot exclude dz = 0; a true dz = 20 µm
        fits ``dz = 19.68 ± 9.24`` → determinate under ``m = 5`` (zero
        excluded at 2.1σ). The Jacobian error bar is used *as-is* — no
        residual-variance scaling, which would rescale every fit's error
        bar (including pinned ones) and is not a trivial change. An
        indeterminate result carries ``dz_um = NaN`` and
        ``indeterminate = True``.

    The observed spectrum of ``target`` is recomputed on the same default
    binning; the model MTF² is evaluated on the target's own sampling grid
    so model and data share one DFT convention. The fit is a coarse
    dz-profile scan (slope solved in closed form per dz) for a starting
    point, refined by :func:`scipy.optimize.least_squares` over
    ``(dz, slope)`` with ``dz ∈ [0, 120]`` µm.
    """
    target = np.asarray(target, dtype=np.float64)
    freqs = np.asarray(freqs, dtype=np.float64)
    s_obj = np.asarray(object_spectrum, dtype=np.float64)
    if not (s_obj.size == freqs.size):
        raise ValueError(
            f"estimate_dz: object_spectrum ({s_obj.size}) and freqs "
            f"({freqs.size}) must have the same length"
        )
    bins = radial_bins(target.shape, n_bins=freqs.size)
    _, s_obs_raw = radial_spectrum(target, bins)

    sigma2 = float(noise_sigma) ** 2
    # debias: subtract the known noise power so the ratio estimates
    # |MTF|² rather than |MTF|² + sigma²/S_obj (the additive floor
    # flattens the decay and biases dz low near the knee)
    s_obs = np.where(np.isfinite(s_obs_raw), s_obs_raw - sigma2, np.nan)
    pos = np.isfinite(s_obs) & np.isfinite(s_obj) & (s_obs > 0) & (s_obj > 0)
    # knee/horizon: highest frequency whose observed power still exceeds
    # KNEE_SNR × the noise power, i.e. residual signal >= (KNEE_SNR-1)·sigma²
    # (equivalently R > noise_floor = sigma2/S_obj with margin)
    above = np.isfinite(s_obs_raw) & (s_obs_raw > KNEE_SNR * sigma2)
    f_horizon = float(freqs[above].max()) if above.any() else 0.0

    band = pos & (freqs >= F_LO) & (freqs <= f_horizon)
    unrecoverable = bool(f_horizon < F_LO or band.sum() < MIN_BAND_BINS)
    if unrecoverable:
        # no fit possible: NaN estimates flag "no opinion" downstream
        return DzResult(dz_um=float("nan"), dz_err_um=float("nan"),
                        slope=float("nan"), f_horizon=f_horizon,
                        unrecoverable=True, haze_dominated=False)

    f_band = freqs[band]
    y = np.log(s_obs[band] / s_obj[band])
    grid_size = int(max(target.shape))

    # --- coarse dz-profile scan, slope optimal in closed form per dz ---
    lfb = np.log(f_band)
    denom = float(lfb @ lfb)
    best_sse, best = np.inf, (0.5 * DZ_MAX, 0.0)
    for dz in np.linspace(0.0, DZ_MAX, 25):
        m2 = np.clip(mtf2_radial(dz, channel, f_band, grid_size=grid_size),
                     _MTF2_FLOOR, None)
        m = np.log(m2)
        slope = float((y - m) @ lfb / denom) if denom > 0 else 0.0
        resid = y - m - slope * lfb
        sse = float(resid @ resid)
        if sse < best_sse:
            best_sse, best = sse, (float(dz), slope)

    # --- joint (dz, slope) refinement ---
    res = least_squares(
        _ratio_model_residuals, best, args=(f_band, y, channel, grid_size),
        bounds=([0.0, -np.inf], [DZ_MAX, np.inf]), diff_step=1e-4,
    )
    dz, slope = float(res.x[0]), float(res.x[1])
    resid = np.asarray(res.fun)
    rms = float(np.sqrt(np.mean(resid**2)))
    try:
        jac = np.asarray(res.jac)
        cov = np.linalg.inv(jac.T @ jac)
        dz_err = float(np.sqrt(max(cov[0, 0], 0.0)))
    except np.linalg.LinAlgError:
        dz_err = float("inf")

    # haze: the ratio must decrease with frequency like an MTF; proxies for
    # "not monotone decreasing" are a rising high-f half, a power-law slope
    # that dominates the MTF shape, or residuals the model cannot explain
    lo_half = float(np.median(y[: y.size // 2]))
    hi_half = float(np.median(y[(y.size + 1) // 2:]))
    rising = hi_half > lo_half
    haze_dominated = bool(
        rising or abs(slope) > HAZE_MAX_SLOPE or rms > HAZE_MAX_RMS
    )

    # indeterminacy ("no opinion") path — see the min_dz_identification
    # docstring above for the criterion and its measured examples. The
    # forced two-parameter fit always returns *a* dz; when its error bar
    # cannot exclude dz = 0 the number is noise, not signal, and nan is
    # the honest value (measured on W042 round 3: every recoverable patch
    # pinned at the 0 bound with a degenerate dz_err).
    indeterminate = False
    if min_dz_identification > 0.0:
        if np.isfinite(dz_err):
            indeterminate = bool(dz_err >= min_dz_identification
                                 and abs(dz) < dz_err)
        else:  # inf/nan: singular Jacobian -> no usable error bar at all
            indeterminate = True
    if indeterminate:
        dz = float("nan")

    return DzResult(dz_um=dz, dz_err_um=dz_err, slope=slope,
                    f_horizon=f_horizon, unrecoverable=False,
                    haze_dominated=haze_dominated,
                    indeterminate=indeterminate)


def sharpness(tile: np.ndarray) -> float:
    """Sharpness of a tile: variance of the Laplacian of the L1-normalized tile.

    The tile is normalized by its mean absolute value (its L1 scale), so
    the returned number is dimensionless and comparable across tiles. The
    Laplacian kills any constant offset, so background level does not
    affect the score; blur suppresses high frequencies and lowers it.
    Flat tiles score 0.0.
    """
    x = np.asarray(tile, dtype=np.float64)
    scale = float(np.abs(x).mean())
    if scale == 0.0:
        return 0.0
    return float(np.var(laplace(x / scale)))


def select_references(tiles: list[np.ndarray], n_tiles: int = 3,
                      patch: int = 512) -> np.ndarray:
    """Pick reference patches: the sharpest ``n_tiles`` tiles, best window each.

    Tiles are ranked by :func:`sharpness` of the full tile; each of the
    ``n_tiles`` sharpest tiles contributes its sharpest ``patch``-sized
    window (scanned on a ``patch // 2`` stride). Returns a single
    ``(patch, patch)`` array when one tile is selected, else a stacked
    ``(n, patch, patch)`` array. Tiles smaller than ``patch`` are skipped;
    raises :class:`ValueError` if no tile can supply a window.
    """
    cands = []  # (score, tile index)
    for idx, tile in enumerate(tiles):
        tile = np.asarray(tile)
        if tile.shape[0] < patch or tile.shape[1] < patch:
            continue
        cands.append((sharpness(tile), idx))
    if not cands:
        raise ValueError(
            f"select_references: no tile is at least {patch}x{patch}"
        )
    cands.sort(key=lambda c: -c[0])
    step = max(1, patch // 2)
    patches = []
    for _, idx in cands[: max(1, n_tiles)]:
        tile = np.asarray(tiles[idx])
        best = None
        for y in range(0, tile.shape[0] - patch + 1, step):
            for x in range(0, tile.shape[1] - patch + 1, step):
                window = tile[y:y + patch, x:x + patch]
                score = sharpness(window)
                if best is None or score > best[0]:
                    best = (score, window)
        patches.append(best[1])
    stack = np.stack(patches)
    return stack[0] if stack.shape[0] == 1 else stack


def build_object_spectrum(ref_patches: list[np.ndarray], bins) -> np.ndarray:
    """Median radial power spectrum over in-focus reference patches.

    ``bins`` comes from :func:`radial_bins` for the (common) patch shape.
    Per-bin powers of :func:`radial_spectrum` are combined with the median
    across patches (robust against an off patch); bins that are empty in
    every patch stay NaN.
    """
    powers = [radial_spectrum(np.asarray(p), bins)[1] for p in ref_patches]
    with np.errstate(invalid="ignore"):
        return np.nanmedian(np.stack(powers), axis=0)
