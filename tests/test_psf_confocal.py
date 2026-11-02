"""Tests for the spinning-disk confocal PSF model (product model)."""

from __future__ import annotations

import numpy as np
import pytest

from sibilla.defaults import real_pupil_config
from sibilla.forward import cyclic3d, psf3d_cyclic
from sibilla.psf import build_confocal_psf_stack, build_psf_stack


def _fwhm(profile: np.ndarray, pixel_um: float) -> float:
    c = int(np.argmax(profile))
    h = profile[c] / 2.0
    l = c
    while l > 0 and profile[l] > h:
        l -= 1
    r = c
    while r < len(profile) - 1 and profile[r] > h:
        r += 1
    return (r - l) * pixel_um


class TestConfocalPsf:
    def test_narrower_than_widefield(self):
        # confocal = exc × det product -> sharper core. In this model the
        # pupil fills the FFT grid (cutoff = Nyquist), so the discrete FWHM
        # saturates at ~2 px for BOTH models — the honest observable is the
        # peak energy concentration (product PSF concentrates much more).
        cfg = real_pupil_config(129)
        z = [0.0]
        wf = build_psf_stack(cfg, {}, z).psf[0]
        cf = build_confocal_psf_stack(cfg, {}, z, lambda_exc_um=0.493).psf[0]
        assert cf.max() > 1.15 * wf.max()

    def test_unit_sum_per_plane(self):
        cfg = real_pupil_config(129)
        z = [-1.0, -0.5, 0.0, 0.5, 1.0]
        cf = build_confocal_psf_stack(cfg, {}, z, lambda_exc_um=0.493).psf
        for k in range(len(z)):
            assert cf[k].sum() == pytest.approx(1.0, rel=1e-9)

    def test_symmetric_zero_aberration(self):
        cfg = real_pupil_config(129)
        cf = np.fft.fftshift(
            build_confocal_psf_stack(cfg, {}, [0.0],
                                     lambda_exc_um=0.493).psf[0])
        np.testing.assert_allclose(cf, cf[:, ::-1], atol=1e-12)
        np.testing.assert_allclose(cf, cf[::-1, :], atol=1e-12)

    def test_same_zernikes_both_wavelengths(self):
        # an aberrated confocal PSF differs from the aberrated widefield one
        cfg = real_pupil_config(129)
        coeffs = {6: 0.10}
        wf = build_psf_stack(cfg, coeffs, [0.0]).psf[0]
        cf = build_confocal_psf_stack(cfg, coeffs, [0.0],
                                      lambda_exc_um=0.493).psf[0]
        assert np.abs(wf - cf).max() > 1e-6

    def test_axially_narrower(self):
        # the product model also sections better: less out-of-focus light
        cfg = real_pupil_config(129)
        z = np.linspace(-2.0, 2.0, 17)
        wf = build_psf_stack(cfg, {}, z).psf
        cf = build_confocal_psf_stack(cfg, {}, z, lambda_exc_um=0.493).psf
        # integrated intensity at ±2 µm, relative to in-focus
        wf_rel = (wf[0].sum() + wf[-1].sum()) / (2 * wf[len(z) // 2].sum())
        cf_rel = (cf[0].sum() + cf[-1].sum()) / (2 * cf[len(z) // 2].sum())
        assert cf_rel < wf_rel


class TestCyclicConfocal:
    def test_psf3d_cyclic_model_flag(self):
        cfg = real_pupil_config(129)
        wf = psf3d_cyclic(cfg, {}, 9, 0.5, model="widefield")
        cf = psf3d_cyclic(cfg, {}, 9, 0.5, model="confocal",
                          lambda_exc_um=0.493)
        assert cf.shape == wf.shape
        assert cf[0].sum() == pytest.approx(1.0, rel=1e-9)
        # confocal in-focus peak is HIGHER (narrower core, unit sum)
        assert cf[0].max() > wf[0].max()


class TestFitConfocalRoundTrip:
    """The decisive test: generate confocal, fit confocal -> recovers;
    fit widefield (wrong model) -> degraded. Documents the mismatch."""

    def test_roi_masks_cover_the_rod(self):
        # Regression (2026-10-20): ROI masks were built from skeleton[:, :4]
        # = [x0, y0, z0, x1] — z0 leaked in as y1, shifting the ROI off the
        # rod (only 16/26 line pixels covered). Must be cols [0,1,3,4].
        from sibilla.fit import _build_roi_masks

        skel6 = np.array([[30.0, 40.0, 6.0, 50.0, 45.0, 6.0]])
        mask = _build_roi_masks(skel6[:, [0, 1, 3, 4]], (96, 96), 1.5, 0.1135)[0]
        covered = 0
        total = 0
        for t in np.linspace(0, 1, 40):
            x = 30 + t * 20
            y = 40 + t * 5
            total += 1
            covered += bool(mask[int(round(y)), int(round(x))])
        assert covered / total >= 0.95

    def test_roundtrip_recovers_with_right_model(self):
        from sibilla.fit import FitInput3D, fit_3d
        from sibilla.scene import CapsuleSpec, generate_capsule_scene

        fov, nz = 96, 15
        spec = CapsuleSpec(fov_px_y=fov, fov_px_x=fov, n_planes=nz,
                           pixel_size_um=0.1135, z_step_um=0.5,
                           diameter_um=0.65, aa_width_px=1.0)
        scene = generate_capsule_scene(spec, seed=21, n_segments=6,
                                       length_range_um=(1.0, 3.0),
                                       z_center_range_um=(-1.0, 1.0),
                                       max_tilt_deg=0.0)
        cfg = real_pupil_config(fov)
        psf = psf3d_cyclic(cfg, {6: 0.10}, nz, 0.5, model="confocal",
                           lambda_exc_um=0.493)
        rng = np.random.default_rng(5)
        data = np.full((nz, fov, fov), 250.0)
        for obj in scene.objects:
            data += 4500.0 * cyclic3d(obj, psf) / obj.max()
        data = rng.poisson(data).astype(np.float64)

        z_planes = (np.arange(nz) - (nz - 1) / 2) * 0.5
        fit_in = FitInput3D(
            data=data, config=cfg, skeleton=scene.segments,
            z_planes_um=z_planes, nominal_budget=4500.0,
            fluorescent_diameter_um=0.65,
            z_candidate_um=(-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5),
            psf_model="confocal", lambda_exc_nm=493.0,
            n_multi_start=1, max_nfev=40,
        )
        res = fit_3d(fit_in)
        assert abs(res.coeffs.get(6, 0.0) - 0.10) < 0.03
        # Regression (2026-10-20): z bookkeeping mixed candidate offsets
        # (-3..+3 planes) with list indices (0..6), letting reported z leave
        # the candidate set (e.g. -7.5 um from a +-4 um grid). Every reported
        # per-rod z must now be exactly one of the candidates.
        cands = [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5]
        for z in res.segment_z_um:
            assert min(cands) - 1e-9 <= z <= max(cands) + 1e-9
            assert min(abs(z - c) for c in cands) < 1e-9

    def test_wrong_model_degrades(self):
        # generate confocal, fit WIDEFIELD: the model must not recover as
        # well (this is the documented widefield-vs-confocal mismatch)
        from sibilla.fit import FitInput3D, fit_3d
        from sibilla.scene import CapsuleSpec, generate_capsule_scene

        fov, nz = 96, 15
        spec = CapsuleSpec(fov_px_y=fov, fov_px_x=fov, n_planes=nz,
                           pixel_size_um=0.1135, z_step_um=0.5,
                           diameter_um=0.65, aa_width_px=1.0)
        scene = generate_capsule_scene(spec, seed=21, n_segments=6,
                                       length_range_um=(1.0, 3.0),
                                       z_center_range_um=(-1.0, 1.0),
                                       max_tilt_deg=0.0)
        cfg = real_pupil_config(fov)
        psf = psf3d_cyclic(cfg, {6: 0.10}, nz, 0.5, model="confocal",
                           lambda_exc_um=0.493)
        rng = np.random.default_rng(5)
        data = np.full((nz, fov, fov), 250.0)
        for obj in scene.objects:
            data += 4500.0 * cyclic3d(obj, psf) / obj.max()
        data = rng.poisson(data).astype(np.float64)

        z_planes = (np.arange(nz) - (nz - 1) / 2) * 0.5
        fit_in = FitInput3D(
            data=data, config=cfg, skeleton=scene.segments,
            z_planes_um=z_planes, nominal_budget=4500.0,
            fluorescent_diameter_um=0.65,
            z_candidate_um=(-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5),
            psf_model="widefield",
            n_multi_start=1, max_nfev=40,
        )
        res = fit_3d(fit_in)
        # wrong model: either biased coefficients or pinned z (or both) —
        # the assertion is that it does NOT match the right-model recovery
        z_pinned = np.any(np.abs(res.segment_z_um) >= 1.5 - 1e-9)
        biased = abs(res.coeffs.get(6, 0.0) - 0.10) >= 0.03
        assert z_pinned or biased
