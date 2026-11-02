"""3D Richardson–Lucy deconvolution of a whole CZI channel (spec §3).

- `build_psf_volume`: the Zernike-derived 3D PSF at the data's z-step,
  centered in a (Nz, Ny, Nx) volume (PSF grid uses the XY pixel size;
  z spacing comes from the z-planes argument — see the docstring).
- `deconvolve_volume`: per-z-slab RL (overlap = PSF z-extent), blended in
  the overlaps. Uses the data's own z-step so the PSF is sampled on the
  same grid as the volume.
- Per-plane background subtraction before RL (spec §3/§6).
- Channel guard: only fluorescence channels are deconvolved (Review Focus 5).

Usage:
    python -m sibilla.deconvolve --czi GoC-Exp4_C3_7.czi --channel 1 \
        --psf psf.json --out deconvolved_ch1.tif [--n-iter 30]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .defaults import REAL_CHANNEL_LAMBDA_NM, real_pupil_config
from .io import load_channel
from .psf import PupilConfig, build_confocal_psf_stack, build_psf_stack

#: z-extent of the widefield PSF halo (µm) — sets slab overlap (spec §3).
PSF_Z_EXTENT_UM = 3.0


def assert_fluorescence_channel(channel: int) -> None:
    """Refuse non-fluorescence channels (Brightfield/reflected — Review Focus 5)."""
    if channel not in REAL_CHANNEL_LAMBDA_NM:
        raise SystemExit(
            f"channel {channel} is not a fluorescence channel "
            f"(valid: {sorted(REAL_CHANNEL_LAMBDA_NM)}) — refusing to "
            f"deconvolve with an epifluorescence PSF")


def build_psf_volume(config: PupilConfig, coeffs: dict[int, float],
                     z_step_um: float,
                     shape: tuple[int, int, int] = (13, 96, 96),
                     model: str = "widefield",
                     lambda_exc_nm: float = 493.0,
                     z_oversample: int = 1,
                     per_plane_kernel: bool = False) -> np.ndarray:
    """Centered 3D PSF volume (unit-sum) at the data's z-step.

    Default shape covers the plan's ±3 µm PSF halo (spec §3 deconvolve row)
    on a 96² XY grid (~10.9 µm at 0.1135 µm px — holds the widest defocused
    ring at ±3 µm). z is centered (in-focus at index (nz-1)//2) and the XY
    peak is fftshifted to center; `deconvolve_volume` passes the volume to
    fftconvolve(mode='same'), which expects this centered layout.

    z_oversample (2026-10-22): each data plane is a z_step_um-thick slab, so
    the physically-correct kernel is the PSF INTEGRATED over each slab. With
    z_oversample>1 we evaluate on a finer z-grid and block-average into slabs
    (z_oversample=1 = point-sample at slab centers, the original behavior).
    A broader, slab-integrated axial kernel is *better conditioned* for RL:
    point-sampling the sub-resolution axial response yields a flatter
    per-plane kernel, which weakens z-localization and lets RL shuffle flux
    between adjacent planes (the bright-slice artifact).

    model: "widefield" or "confocal" (spinning-disk product PSF, same
    Zernikes at excitation λ_exc and detection λ = config.lambda_um).
    """
    nz, ny, nx = shape
    cfg = PupilConfig(pixel_size=config.pixel_size,
                      lambda_um=config.lambda_um, na=config.na,
                      n_im=config.n_im, grid_size=ny)
    if z_oversample > 1:
        z_fine = ((np.arange(nz * z_oversample)
                   - (nz * z_oversample - 1) / 2.0)
                  * z_step_um / z_oversample)
        if model == "confocal":
            # per_plane=False by default: keep the confocal AXIAL ENVELOPE
            # (per-plane energy decays away from focus = the pinhole's
            # sectioning). Re-normalizing per plane erases it and makes
            # RL's z-system rank-deficient -> flux shuffling between
            # planes (2026-10-22 bright-slice fix).
            fine = build_confocal_psf_stack(cfg, coeffs, z_fine,
                                            lambda_exc_um=lambda_exc_nm / 1000.0,
                                            per_plane=per_plane_kernel).psf
        else:
            fine = build_psf_stack(cfg, coeffs, z_fine).psf
        psf = fine.reshape(nz, z_oversample, ny, ny).mean(axis=1)
    else:
        z_planes = (np.arange(nz) - (nz - 1) / 2.0) * z_step_um
        if model == "confocal":
            psf = build_confocal_psf_stack(cfg, coeffs, z_planes,
                                           lambda_exc_um=lambda_exc_nm / 1000.0,
                                           per_plane=per_plane_kernel).psf
        else:
            psf = build_psf_stack(cfg, coeffs, z_planes).psf      # (nz, ny, ny)
    if ny != nx:
        sx = (ny - nx) // 2
        psf = psf[:, :, sx:sx + nx]
    vol = np.fft.fftshift(psf, axes=(1, 2))
    s = vol.sum()
    if s > 0:
        vol /= s
    return vol


def subtract_plane_background(stack: np.ndarray) -> np.ndarray:
    """Per-plane median subtraction, clipped at 0 (spec §3/§6)."""
    stack = np.asarray(stack, dtype=np.float64)
    med = np.median(stack, axis=(1, 2), keepdims=True)
    return np.maximum(stack - med, 0.0)


def richardson_lucy_3d(image: np.ndarray, psf: np.ndarray,
                       n_iter: int = 30, eps: float = 1e-12) -> np.ndarray:
    """3D Richardson–Lucy via `scipy.signal.fftconvolve(mode='same')`.

    Same proven mechanism as the tier-1 `metrics.richardson_lucy`: the PSF is
    a centered kernel (peak at its center voxel), `fftconvolve` handles
    padding/centering, so no manual FFT embedding or grid-match is needed.
    """
    from scipy.signal import fftconvolve

    image = np.maximum(np.asarray(image, dtype=np.float64), 0.0)
    psf = np.asarray(psf, dtype=np.float64)
    # The PSF from build_psf_volume is a CENTERED kernel (peak at the middle
    # voxel). fftconvolve(mode='same') accepts a centered kernel of ANY size
    # relative to the image (it pads internally), so we must NOT crop it —
    # cropping a smaller-than-image PSF truncates/shifts it (the whole-CZI
    # case). Just normalize and use as-is.
    if psf.sum() > 0:
        psf = psf / psf.sum()
    psf_mirror = psf[::-1, ::-1, ::-1]
    est = np.full_like(image, image.mean() if image.size else 0.0)
    for _ in range(n_iter):
        blur = fftconvolve(est, psf, mode="same") + eps
        ratio = image / blur
        est = est * fftconvolve(ratio, psf_mirror, mode="same")
        est = np.maximum(est, 0.0)
    return est


def _worker_rl(slab: np.ndarray, psf: np.ndarray, n_iter: int) -> np.ndarray:
    """Worker entry point: RL-deconvolve one slab (spawn-safe, picklable)."""
    return richardson_lucy_3d(slab, psf, n_iter=n_iter)


def _worker_rl_gpu(args: tuple) -> np.ndarray:
    """GPU worker: sets CUDA_VISIBLE_DEVICES BEFORE importing cupy (fork-safe:
    the parent process never initializes CUDA, so children may)."""
    slab, psf, n_iter, gpu_id = args
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id % 4)
    return richardson_lucy_3d_gpu(slab, psf, n_iter=n_iter)


def richardson_lucy_3d_gpu(image: np.ndarray, psf: np.ndarray,
                           n_iter: int = 30, eps: float = 1e-12,
                           dtype=None) -> np.ndarray:
    """3D RL in Fourier space on the GPU (CuPy).

    Reproduces `richardson_lucy_3d` (scipy.fftconvolve mode='same' with the
    centered kernel) exactly: zero-pad image and kernel (as-is, peak stays
    at its centered index) to a power-of-2 F >= N_img + N_psf - 1 so the
    circular convolution equals the linear one, precompute FFT(kernel) and
    FFT(mirror kernel) once, then per iteration: FFT(est), multiply, IFFT,
    crop [off : off + N_img] with off = (N_k - 1)//2 per axis — which is
    precisely scipy's 'same' alignment. Validated by test_gpu_matches_cpu.
    """
    import cupy as cp

    # float64 default: RL amplifies low-order float error multiplicatively
    # (est <- est * conv(ratio), conv up to O(100) where blur ~ eps), so a
    # float32 trajectory diverges from the float64 CPU path within a few
    # iterations. Memory is ample (full channel ~15 GB/worker in f64 < 48 GB).
    if dtype is None:
        dtype = cp.float64
    nz, m, n = image.shape
    nz_k, ny_k, nx_k = psf.shape
    def _p2(x: int) -> int:
        return 1 << (x - 1).bit_length()
    Fz, Fy, Fx = _p2(nz + nz_k - 1), _p2(m + ny_k - 1), _p2(n + nx_k - 1)
    offz, offy, offx = (nz_k - 1) // 2, (ny_k - 1) // 2, (nx_k - 1) // 2

    img = cp.asarray(image, dtype=dtype)
    psf_c = cp.asarray(psf, dtype=dtype)
    if psf_c.sum() > 0:
        psf_c /= psf_c.sum()

    pad_k = ((0, Fz - nz_k), (0, Fy - ny_k), (0, Fx - nx_k))
    # kernel AS-IS (centered, no ifftshift): circular conv of the padded
    # arrays == linear conv, and scipy's mode='same' is the crop
    # [off : off + N_img].
    K = cp.fft.fftn(cp.pad(psf_c, pad_k))
    Km = cp.fft.fftn(cp.pad(psf_c[::-1, ::-1, ::-1], pad_k))

    crop = (slice(offz, offz + nz), slice(offy, offy + m),
            slice(offx, offx + n))
    est = cp.full((nz, m, n), float(image.mean()) if image.size else 0.0,
                  dtype=dtype)
    work = cp.zeros((Fz, Fy, Fx), dtype=dtype)          # reused scratch
    for _ in range(n_iter):
        work[...] = 0
        work[:nz, :m, :n] = est
        E = cp.fft.fftn(work)
        blur = cp.real(cp.fft.ifftn(E * K))[crop] + eps
        del E
        ratio = img / blur
        del blur
        work[...] = 0
        work[:nz, :m, :n] = ratio
        est = est * cp.real(cp.fft.ifftn(cp.fft.fftn(work) * Km))[crop]
        est = cp.maximum(est, 0.0)
    return cp.asnumpy(est)


def _slab_tiling(nz: int, slab_planes: int, overlap_planes: int,
                 nz_psf: int) -> tuple[int, int, int, int]:
    """(slab, overlap, step, guard) for the trim-to-trusted assembly.

    guard = 2*half_psf: a slab plane within half_psf of a slab edge has its
    PSF tail truncated by the zero-padding -> an RL ratio spike there, and
    the mirror convolution (est *= conv(ratio, K_mirror)) carries that spike
    a further ~half_psf inward, so planes within 2*half_psf of a slab edge
    cannot be trusted. With overlap = 2*guard the kept regions
    [z0+guard, z1-guard) tile the z-axis exactly (step = slab-overlap);
    slab grows to max(slab_planes, 2*overlap) to keep the kept width >= 1.
    For the production 13-plane PSF (half 6) this is slab 48 / step 24 /
    guard 12 — the padded FFT is the same 64-plane size as the old
    slab-24 tiling, and there are FEWER slabs (9 vs 11).

    2026-10-26 z-streak fix: the production tiling previously trimmed only
    guard=3 < 6 = half_psf, leaving the user's thin z-streaks at the seam
    planes (z~73/93, +13% plane-mean spikes vs the monolithic no-slabbing
    run; guard >= 6 trims are clean on the real channel, guard = 2*half
    also on the constant-volume sweep).
    """
    half = (nz_psf - 1) // 2
    if nz <= slab_planes or half == 0:
        return slab_planes, 0, max(1, slab_planes), 0
    guard = 2 * half
    overlap = 2 * guard                        # kept width = slab - 2*guard = step
    slab = max(slab_planes, 2 * overlap)       # step = slab - overlap = overlap
    step = slab - overlap
    return slab, overlap, step, guard


def deconvolve_volume(stack: np.ndarray, psf: np.ndarray, n_iter: int = 30,
                      slab_planes: int = 24,
                      overlap_planes: int = 6,
                      n_workers: int = 1,
                      device: str = "cpu") -> np.ndarray:
    """3D RL on z-slabs with trim-to-trusted assembly (spec §3), parallel.

    `psf` must be a centered (Nz_psf, Ny_psf, Nx_psf) volume; it is
    downsampled/cropped to the volume's XY as needed. The z-axis is tiled
    into slabs (see `_slab_tiling`): `slab_planes` is a minimum slab size,
    the actual overlap/guard are derived from the PSF's z-extent so that no
    kept plane lies within 2*half_psf of a slab edge (zero-padding would
    otherwise truncate the kernel tail and RL streaks the seam — the
    2026-10-26 z-streak fix). Each output plane is taken from exactly one
    slab; no blending.
    With n_workers > 1 the independent slabs run in separate processes
    (fork context: the RL loops use single-threaded pocketfft — no OpenBLAS
    pool — so children inherit no live thread-pool state). device="gpu"
    runs each slab on its own NVIDIA GPU (CUDA_VISIBLE_DEVICES set inside
    the worker before cupy's first import). Results are identical to the
    serial path (test_parallel_matches_serial_exactly;
    test_gpu_matches_cpu for the GPU arm).
    """
    stack = np.asarray(stack, dtype=np.float64)
    nz, m, n = stack.shape
    nz_psf, ny_psf, nx_psf = psf.shape
    # psf is a centered kernel (peak at center) from build_psf_volume. Crop a
    # centered (m, n) window; fftconvolve(mode='same') handles the rest.
    # (2026-10-26) clamp the start at 0: for data WIDER than the PSF grid
    # (all production slabs: 1440 x 1920 vs 96 x 96) the old unclamped
    # `cy - m//2` went negative and numpy's negative-start wrap silently
    # selected a small CORNER of the kernel (e.g. an 8x8 slice) — it only
    # worked in production because the end index clamped to the grid edge.
    cy, cx = ny_psf // 2, nx_psf // 2
    sy, ex = cy - m // 2, cy - m // 2 + m
    sx, exx = cx - n // 2, cx - n // 2 + n
    psf_xy = psf[:, sy:ex, sx:exx]
    if psf_xy.sum() > 0:
        psf_xy = psf_xy / psf_xy.sum()
    # z-streak tiling (see _slab_tiling): trim 2*half_psf from each slab
    # edge so no kept plane carries the zero-padding truncation spike
    # (or its mirror-convolution spread).
    slab, overlap, step, guard = _slab_tiling(nz, slab_planes, overlap_planes,
                                              nz_psf)
    ranges = [(z0, min(z0 + slab, nz))
              for z0 in range(0, nz, step)]

    if device == "gpu":
        # one worker per slab, round-robin over the (up to) 4 GPUs; the
        # parent process never imports cupy (fork-safe).
        args = [(stack[z0:z1], psf_xy, n_iter, i)
                for i, (z0, z1) in enumerate(ranges)]
        if n_workers > 1 and len(ranges) > 1:
            import multiprocessing as mp
            ctx = mp.get_context("fork")
            with ctx.Pool(min(n_workers, len(ranges))) as pool:
                decs = pool.map(_worker_rl_gpu, args)
        else:
            decs = [_worker_rl_gpu(a) for a in args]
    elif n_workers > 1 and len(ranges) > 1:
        import multiprocessing as mp
        # fork (Linux default): safe here because the RL loop is dominated
        # by fftconvolve (pocketfft, single-threaded, no OpenBLAS pool), so
        # children inherit no live BLAS thread-pool mutexes. spawn would
        # re-import __main__ and break under pytest/stdin.
        ctx = mp.get_context("fork")
        n_use = min(n_workers, len(ranges))
        args = [(stack[z0:z1], psf_xy, n_iter) for z0, z1 in ranges]
        with ctx.Pool(n_use) as pool:
            decs = pool.starmap(_worker_rl, args)
    else:
        decs = [richardson_lucy_3d(stack[z0:z1], psf_xy, n_iter=n_iter)
                for z0, z1 in ranges]

    out = np.zeros_like(stack)
    # trim-to-trusted assembly (2026-10-22; guard corrected 2026-10-26):
    # each INTERIOR plane is taken from the slab whose center is nearest to
    # it (i.e. where it lies >= guard from both slab edges); the true volume
    # edges (first/last `guard` planes) are taken from the first/last slab
    # that reaches them. No blending, no clobbering: a later tiny tail slab
    # must never overwrite an already-clean interior plane. Verified against
    # the monolithic no-slabbing run: guard=2*half_psf -> boundary planes
    # match within ~1% (the old guard=3 left the user's z~73/93 streaks).
    filled = np.zeros(nz, dtype=bool)
    for (z0, z1), dec in zip(ranges, decs):
        lo = max(z0 + guard, 0)
        hi = min(z1 - guard, nz)
        if hi > lo:
            out[lo:hi] = dec[lo - z0:hi - z0]
            filled[lo:hi] = True
    # true volume edges: fill any unfilled planes from the nearest slab
    for (z0, z1), dec in zip(ranges, decs):
        for z in range(z0, min(z1, nz)):
            if not filled[z]:
                out[z] = dec[z - z0]
                filled[z] = True
    # (defensive) any still-unfilled plane — only possible if a slab is
    # shorter than guard on both sides and no neighbor covered it — falls
    # back to the raw background-subtracted input.
    if not filled.all():
        out[~filled] = stack[~filled]
    return out


def saturation_fraction(stack: np.ndarray, sat_level: int = 16300) -> float:
    """Fraction of pixels at/above `sat_level` (Review Focus 2)."""
    stack = np.asarray(stack)
    return float((stack >= sat_level).mean())


def write_ome_tiff(path: str | Path, stack: np.ndarray, pixel_um: float,
                   z_step_um: float) -> None:
    """Write a (Nz, M, N) volume as an OME-TIFF, preserving absolute counts.

    Counts are clipped to uint16 without rescaling, so a deconvolved volume
    displays on the same gray scale as the raw data (spec §6). Physical sizes
    go in OME metadata (PhysicalSizeX/Y/Z); `description` is the portable
    note (image_description is not a valid kwarg in current tifffile).
    """
    import tifffile

    s = np.asarray(stack, dtype=np.float64)
    s = np.clip(np.rint(s), 0, 65535).astype(np.uint16)
    note = (f"deconvolved volume; xy={pixel_um}um z={z_step_um}um; "
            f"absolute counts preserved (uint16)")
    try:
        # OME-TIFF with physical sizes (works on tifffile versions that
        # expose these kwargs).
        tifffile.imwrite(str(path), s, axes="ZYX",
                         metadata={"PhysicalSizeX": pixel_um,
                                   "PhysicalSizeY": pixel_um,
                                   "PhysicalSizeZ": z_step_um},
                         description=note)
    except TypeError:
        # Version skew (e.g. server tifffile rejects `axes`): write a plain
        # (Z, Y, X) multi-page TIFF so the data is never lost to metadata
        # kwarg differences. Absolute counts are preserved either way.
        tifffile.imwrite(str(path), s, description=note)
    return


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="3D RL deconvolution of a CZI channel")
    ap.add_argument("--czi", required=True)
    ap.add_argument("--channel", type=int, default=1)
    ap.add_argument("--psf", required=True,
                    help="psf.json from fit_real_crop: "
                         "{coeffs, na, n_im, lambda_nm, pixel_size_um, "
                         "z_step_um, psf_model, lambda_exc_nm}")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-iter", type=int, default=30)
    ap.add_argument("--z0", type=int, default=0)
    ap.add_argument("--z1", type=int, default=-1)
    ap.add_argument("--zero-coeffs", action="store_true",
                    help="use the theoretical diffraction-limited PSF "
                         "(all Zernikes = 0) instead of the fitted coeffs; "
                         "all other psf.json fields (model, wavelengths) "
                         "still apply")
    ap.add_argument("--n-workers", type=int, default=1,
                    help="parallel z-slab workers (each an independent "
                         "process; a 196-plane channel has ~11 slabs)")
    ap.add_argument("--device", choices=["cpu", "gpu"], default="cpu",
                    help="rl device: 'gpu' runs each z-slab on its own "
                         "NVIDIA GPU (round-robin over up to 4), one "
                         "worker per slab by default")
    ap.add_argument("--flat-z-kernel", action="store_true",
                    help="re-normalize the confocal kernel per z-plane "
                         "(the old behavior that erased the axial "
                         "envelope and caused RL flux shuffling between "
                         "planes — kept only for A/B comparison)")
    args = ap.parse_args(argv)

    assert_fluorescence_channel(args.channel)
    if args.device == "gpu" and args.n_workers == 1:
        # One worker per GPU (4 A40s). Each slab's padded FFT peaks ~19 GB,
        # so 1/GPU is the safe default; with 11 slabs the pool distributes
        # them round-robin (worker k gets slabs k, k+4, k+8 — all GPU k%4).
        args.n_workers = 4
    meta = json.loads(Path(args.psf).read_text())
    cfg = PupilConfig(pixel_size=float(meta["pixel_size_um"]),
                      lambda_um=float(meta["lambda_nm"]) / 1000.0,
                      na=float(meta["na"]), n_im=float(meta["n_im"]),
                      grid_size=96)
    coeffs = {int(k): float(v) for k, v in meta["coeffs"].items()}
    if args.zero_coeffs:
        coeffs = {k: 0.0 for k in coeffs}
    z_step = float(meta["z_step_um"])
    model = meta.get("psf_model", "widefield")
    lam_exc = float(meta.get("lambda_exc_nm", 493.0))
    # 96² XY grid matches the fitter's PSF kernel; 13 planes = ±3 µm halo.
    # Default (no flag): keep the confocal axial envelope (per_plane=False)
    # — the 2026-10-22 bright-slice fix. --flat-z-kernel restores the old
    # per-plane-renormalized kernel for A/B comparison. NOTE: pass the flag
    # value straight through (an earlier revision had an extra `not` here,
    # which silently swapped the two kernel variants — caught by the
    # matrix run's digit-for-digit match with the old flat-12it output).
    vol = build_psf_volume(cfg, coeffs, z_step, shape=(13, 96, 96),
                           model=model, lambda_exc_nm=lam_exc,
                           per_plane_kernel=args.flat_z_kernel)

    stack = load_channel(args.czi, args.channel)
    z1 = args.z1 if args.z1 >= 0 else stack.shape[0]
    slab = stack[args.z0:z1].astype(np.float64)
    slab = subtract_plane_background(slab)
    dec = deconvolve_volume(slab, vol, n_iter=args.n_iter,
                            n_workers=args.n_workers,
                            device=args.device)
    write_ome_tiff(args.out, dec, cfg.pixel_size, z_step)
    tag = "theoretical(zero)" if args.zero_coeffs else "fitted"
    print(f"wrote {args.out}  ({dec.shape}, z {args.z0}..{z1}, "
          f"model={model} psf={tag}, device={args.device}, "
          f"workers={args.n_workers}, "
          f"sat_frac={saturation_fraction(stack[args.z0:z1]):.4f})")


if __name__ == "__main__":
    main()
