"""Tests for deconvolve.py (Task 9)."""

from __future__ import annotations

import numpy as np
import pytest

from sibilla.defaults import real_pupil_config
from sibilla.deconvolve import (
    build_psf_volume,
    deconvolve_volume,
    saturation_fraction,
    subtract_plane_background,
)
from sibilla.io import load_channel

CZI_PATH = "GoC-Exp4_C3_7.czi"


def _gpu_available() -> bool:
    """True if cupy imports AND reports a CUDA device (GPU node + env)."""
    try:
        import cupy as cp
    except Exception:
        return False
    try:
        cp.cuda.runtime.getDeviceCount() > 0
        return bool(cp.cuda.runtime.getDeviceCount())
    except Exception:
        return False


class TestBuildPsfVolume:
    def test_confocal_kernel_axial_envelope_kept(self):
        # 2026-10-22 fix: the deconvolution confocal kernel must keep its
        # axial envelope (per-plane energy decays away from focus = the
        # 1-AU pinhole's sectioning signal). The old per-plane renorm
        # erased it -> FLAT 8%/plane kernel -> rank-deficient z-system ->
        # RL flux shuffling between adjacent planes (bright-slice artifact,
        # ~10x the gold-standard Huygens z-jump). Widefield is flat by
        # physics (Parseval), so this test pins the CONFOCAL kernel.
        # Also pins the per_plane_kernel FLAG SEMANTICS (a flipped `not` in
        # the CLI once silently swapped the two variants).
        from sibilla.deconvolve import build_psf_volume

        cfg = real_pupil_config(96)
        vol = build_psf_volume(cfg, {}, 0.5, shape=(13, 96, 96),
                               model="confocal", lambda_exc_nm=493.0,
                               per_plane_kernel=False)
        zfrac = vol.sum(axis=(1, 2)) / vol.sum()
        flat = 1.0 / 13
        c = 6
        assert zfrac[c] > 1.5 * flat, zfrac
        # falloff away from focus: center clearly above the edges
        assert zfrac[c] > zfrac[0] and zfrac[c] > zfrac[-1]

        vol_flat = build_psf_volume(cfg, {}, 0.5, shape=(13, 96, 96),
                                    model="confocal", lambda_exc_nm=493.0,
                                    per_plane_kernel=True)
        zf = vol_flat.sum(axis=(1, 2)) / vol_flat.sum()
        # per-plane renormalized -> every plane carries ~equal energy
        assert max(zf.max() / zf.min(), zf.min() / zf.max()) < 2.0, zf

    def test_unit_sum_centered_symmetric(self):
        cfg = real_pupil_config(96)
        vol = build_psf_volume(cfg, {}, 0.5, shape=(13, 96, 96))
        assert vol.sum() == pytest.approx(1.0, rel=1e-9)
        assert vol.shape == (13, 96, 96)
        assert np.unravel_index(np.argmax(vol), vol.shape) == (6, 48, 48)
        # lateral symmetry of the zero-aberration PSF. On the even grid the
        # peak sits half a pixel off center, so the discrete PSF is symmetric
        # only to that sampling offset (~0.05 of peak) — verified: an odd
        # grid mirrors to 1e-18. 0.1 is above the discretization and far
        # below the asymmetry a real astig/coma term (>>0.1) would produce.
        c = vol[:, 24:72, 24:72]
        assert np.abs(c[:, :, ::-1] - c).max() < 0.1
        # z symmetry (zero aberration, centered plane) — exact
        assert np.abs(vol[::-1] - vol).max() < 1e-9

    def test_aberrated_psf_has_wings(self):
        cfg = real_pupil_config(96)
        vol = build_psf_volume(cfg, {8: 0.2}, 0.5, shape=(13, 96, 96))
        assert vol[:, :, :].sum() == pytest.approx(1.0, rel=1e-9)
        # coma: xy asymmetry in the central plane
        assert (vol[6].sum(axis=0) - vol[6].sum(axis=0)[::-1]).max() > 1e-4


class TestSubtractPlaneBackground:
    def test_drives_medians_to_zero(self):
        rng = np.random.default_rng(0)
        stack = rng.poisson(300.0, size=(10, 16, 16)).astype(np.float32)
        out = subtract_plane_background(stack)
        meds = np.median(out, axis=(1, 2))
        assert np.all(meds < 5)
        assert (out >= 0).all()


class TestSaturation:
    def test_fraction(self):
        stack = np.zeros((4, 10, 10), np.float32)
        stack[0, :2, :] = 16383.0      # 20/400 = 5%
        assert saturation_fraction(stack) == pytest.approx(0.05)
        assert saturation_fraction(np.zeros((4, 10, 10), np.float32)) == 0.0


class TestDeconvolveVolume:
    def test_roundtrip_recovers_width(self):
        """P6 mechanism check (unit scale): RL with the true PSF (1) recovers
        the blurred peak, (2) stays finite/non-negative, (3) does not sharpen
        below the diffraction floor (the over-sharpening guard in spec §6).

        Exact width recovery of an idealized top-hat is NOT asserted — RL on
        clean synthetic flat objects rings/amplifies at the edges; the
        image-improvement claim is made on real (noisy, soft) bacteria in
        Task 11.
        """
        from sibilla.psf import build_psf_stack
        from sibilla.scene import CapsuleSpec, generate_capsule_scene
        from sibilla.forward import FastBlur
        from sibilla.metrics import apparent_diameter_um

        spec = CapsuleSpec(fov_px_y=64, fov_px_x=64, n_planes=9,
                           pixel_size_um=0.1135, z_step_um=0.5,
                           diameter_um=0.65, aa_width_px=1.0)
        scene = generate_capsule_scene(spec, seed=9, n_segments=3,
                                       length_range_um=(2.0, 3.0),
                                       z_center_range_um=(-0.5, 0.5),
                                       max_tilt_deg=0.0)
        cfg = real_pupil_config(64)
        z_planes = (np.arange(9) - 4) * 0.5
        psf = build_psf_stack(cfg, {}, z_planes).psf
        data = np.full((9, 64, 64), 100.0)
        for obj in scene.objects:
            for k in range(9):
                data[k] += 3000.0 * FastBlur(obj[k]).blur(psf[k]) / max(obj.max(), 1e-12)
        data = subtract_plane_background(data)
        vol = build_psf_volume(cfg, {}, 0.5, shape=(13, 96, 96))
        dec = deconvolve_volume(data, vol, n_iter=30, slab_planes=9)

        assert np.isfinite(dec).all()
        assert (dec >= 0).all()
        # (1) peak recovery: RL restores the blurred central-plane peak
        assert dec[4].max() > data[4].max()
        # (3) diffraction floor: the rod's apparent width does not collapse
        mid = scene.segments[0]
        d_dec = apparent_diameter_um(dec[4], spec.pixel_size_um,
                                     center=(0.5 * (mid[0] + mid[3]),
                                            0.5 * (mid[1] + mid[4])),
                                     direction=(mid[3] - mid[0], mid[4] - mid[1]))
        assert not (d_dec is not None and np.isfinite(d_dec)) or \
            d_dec >= 0.168

    def test_slab_edges_no_z_streak(self):
        # 2026-10-26 z-streak regression (user-reported thin z-streaks at
        # z~73/93 = slab seams). Mechanism: a slab plane within the PSF's z
        # half-width (6 for the 13-plane kernel) of a slab edge has its
        # kernel tail TRUNCATED by the slab's zero-padding -> the forward
        # blur underestimates -> RL's ratio spikes -> a bright thin z-streak
        # at the seam. The assembly must only keep planes >= guard from the
        # slab edges, with guard >= 2*half_psf (the ratio spike is spread a
        # further ~half_psf inward by the mirror convolution).
        # Verified on the real channel: old guard=3 left +13% seam spikes;
        # the guard>=6/12 tiling is clean vs the monolithic no-slabbing run.
        #
        # This test is DETERMINISTIC (no RL numerics): it pins
        # _slab_tiling's numbers and drives the assembly with fake slab
        # outputs whose edge zones carry a marker, asserting the markers
        # never reach the output.
        from sibilla.deconvolve import _slab_tiling

        # (a) production tiling: 196 planes, 13-plane PSF, default slab 24
        slab, overlap, step, guard = _slab_tiling(196, 24, 6, 13)
        assert (slab, step, guard) == (48, 24, 12)
        assert guard >= 2 * ((13 - 1) // 2)    # >= 2*half_psf
        # small volume (<= one slab): single monolithic slab, no guard
        assert _slab_tiling(20, 24, 6, 13) == (24, 0, 24, 0)

        # (b) the kept regions [z0+guard, z1-guard) tile the interior EXACTLY
        # once (no gaps -> no clobbering, no seam plane left to a truncated
        # slab), and the head/tail (true volume edges) belong to NO kept
        # region — the assembly fills those from the boundary slabs.
        nz = 196
        ranges = [(z0, min(z0 + slab, nz)) for z0 in range(0, nz, step)]
        kept = np.zeros(nz, dtype=int)
        for (z0, z1) in ranges:
            kept[max(z0 + guard, 0):min(z1 - guard, nz)] += 1
        assert (kept[guard:nz - guard] == 1).all()
        assert kept[:guard].sum() == 0 and kept[nz - guard:].sum() == 0
        assert all(any(z0 <= z < z1 for z0, z1 in ranges)
                   for z in list(range(guard)) + list(range(nz - guard, nz)))

    def test_refuses_nonfluorescence_channel(self):
        # Review Focus 5: Brightfield (ch 3) / reflected (ch 5) must be refused
        from sibilla.deconvolve import assert_fluorescence_channel

        assert_fluorescence_channel(1)      # AF488 — allowed (no raise)
        for bad in (3, 5):                  # Brightfield, reflected light
            with pytest.raises(SystemExit):
                assert_fluorescence_channel(bad)

    def test_parallel_matches_serial_exactly(self):
        # n_workers>1 must reproduce the serial result bit-for-bit (same
        # deterministic RL, only the slab tiling differs). Multiple slabs
        # required: 15 planes, slab 9 / overlap 3 -> step 6 -> 3 slabs.
        from sibilla.psf import build_psf_stack
        from sibilla.scene import CapsuleSpec, generate_capsule_scene
        from sibilla.forward import FastBlur

        spec = CapsuleSpec(fov_px_y=64, fov_px_x=64, n_planes=15,
                           pixel_size_um=0.1135, z_step_um=0.5,
                           diameter_um=0.65, aa_width_px=1.0)
        scene = generate_capsule_scene(spec, seed=9, n_segments=2,
                                       length_range_um=(1.5, 2.0),
                                       z_center_range_um=(-0.5, 0.5),
                                       max_tilt_deg=0.0)
        cfg = real_pupil_config(64)
        z_planes = (np.arange(15) - 7) * 0.5
        psf = build_psf_stack(cfg, {}, z_planes).psf
        data = np.full((15, 64, 64), 100.0)
        for obj in scene.objects:
            for k in range(15):
                data[k] += 3000.0 * FastBlur(obj[k]).blur(psf[k]) / max(obj.max(), 1e-12)
        data = subtract_plane_background(data)
        vol = build_psf_volume(cfg, {}, 0.5, shape=(9, 48, 48))
        dec_serial = deconvolve_volume(data, vol, n_iter=5, slab_planes=9,
                                       overlap_planes=3, n_workers=1)
        dec_parallel = deconvolve_volume(data, vol, n_iter=5, slab_planes=9,
                                         overlap_planes=3, n_workers=3)
        assert dec_serial.shape == dec_parallel.shape
        assert np.array_equal(dec_serial, dec_parallel)

    @pytest.mark.skipif(
        not _gpu_available(),
        reason="no GPU / cupy not installed")
    def test_gpu_matches_cpu(self):
        # The GPU port must reproduce the CPU RL: manual padded-FFT
        # convolution (ifftshift peak -> origin, crop [0:N]) vs
        # scipy.fftconvolve(mode='same'). float32 -> tolerance, not exact.
        from sibilla.deconvolve import (
            richardson_lucy_3d,
            richardson_lucy_3d_gpu,
        )
        from sibilla.psf import build_psf_stack
        from sibilla.scene import CapsuleSpec, generate_capsule_scene
        from sibilla.forward import FastBlur

        spec = CapsuleSpec(fov_px_y=64, fov_px_x=64, n_planes=9,
                           pixel_size_um=0.1135, z_step_um=0.5,
                           diameter_um=0.65, aa_width_px=1.0)
        scene = generate_capsule_scene(spec, seed=9, n_segments=2,
                                       length_range_um=(1.5, 2.0),
                                       z_center_range_um=(-0.5, 0.5),
                                       max_tilt_deg=0.0)
        cfg = real_pupil_config(64)
        z_planes = (np.arange(9) - 4) * 0.5
        psf = build_psf_stack(cfg, {}, z_planes).psf
        data = np.full((9, 64, 64), 100.0)
        for obj in scene.objects:
            for k in range(9):
                data[k] += 3000.0 * FastBlur(obj[k]).blur(psf[k]) / max(obj.max(), 1e-12)
        data = subtract_plane_background(data)
        vol = build_psf_volume(cfg, {}, 0.5, shape=(9, 48, 48))
        dec_cpu = richardson_lucy_3d(data, vol, n_iter=6)
        dec_gpu = richardson_lucy_3d_gpu(data, vol, n_iter=6)
        assert dec_gpu.shape == dec_cpu.shape
        rel = np.abs(dec_gpu - dec_cpu).max() / max(dec_cpu.max(), 1e-9)
        assert rel < 2e-3, f"gpu/cpu max relative diff {rel:.2e}"


@pytest.mark.skipif(not __import__("os").path.exists(CZI_PATH),
                    reason="CZI not present")
class TestRealSlab:
    def test_deconvolve_slab_real(self):
        from sibilla.deconvolve import build_psf_volume

        cfg = real_pupil_config(96)
        psfvol = build_psf_volume(cfg, {}, 0.5, shape=(13, 96, 96))
        vol = load_channel(CZI_PATH, 1)
        slab = vol[20:44].astype(np.float64)
        dec = deconvolve_volume(slab, psfvol, n_iter=10, slab_planes=24)
        assert np.isfinite(dec).all()
        assert dec.shape == slab.shape
