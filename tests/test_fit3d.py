"""Tests for the 3D fitter (Tasks 5–6)."""

from __future__ import annotations

import numpy as np
import pytest

from sibilla.defaults import BACTERIA_SCENE, real_pupil_config
from sibilla.fit import FitInput, fit
from sibilla.forward import forward_stack
from sibilla.psf import build_psf_stack
from sibilla.scene import generate_scene


def _forward_planar(scene, coeffs, peak, bg_per_plane=None, seed=0):
    """Render a planar scene through the PSF at given coeffs + per-plane bg."""
    cfg = real_pupil_config(scene.spec.fov_px)
    z_planes = (np.arange(scene.spec.n_planes)
                - (scene.spec.n_planes - 1) / 2) * scene.spec.z_step_um
    psf = build_psf_stack(cfg, coeffs, z_planes)
    rng = np.random.default_rng(seed)
    planes = []
    for k, z in enumerate(z_planes):
        img = np.zeros((scene.spec.fov_px, scene.spec.fov_px))
        for o in scene.objects:
            img += o * peak
        planes.append(img)
    obj = planes[0]
    out = forward_stack(obj, psf, amplitude=1.0, background=0.0)
    if bg_per_plane is not None:
        out = out + np.asarray(bg_per_plane)[:, None, None]
    return rng.poisson(out).astype(np.float64), psf, z_planes


class TestLinearModel3D:
    def test_exact_toy_recovery(self):
        from sibilla.fit import _LinearModel3D
        from sibilla.forward import FastBlur

        rng = np.random.default_rng(0)
        nz, m, n = 5, 32, 32
        # two objects, known PSF (gaussian blur kernel), known per-plane bg
        obj1 = np.zeros((m, n)); obj1[10:14, 8:12] = 0.25
        obj2 = np.zeros((m, n)); obj2[20:24, 20:24] = 0.25
        psf = np.zeros((nz, m, n))
        yy, xx = np.mgrid[0:m, 0:n]
        g = np.exp(-((yy - 16) ** 2 + (xx - 16) ** 2) / 8.0)
        for k in range(nz):
            psf[k] = g / g.sum()
        amps_true = np.array([300.0, 700.0])
        bg_true = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        data = np.empty((nz, m, n))
        fb1, fb2 = FastBlur(obj1), FastBlur(obj2)
        for k in range(nz):
            data[k] = amps_true[0] * fb1.blur(psf[k]) + \
                      amps_true[1] * fb2.blur(psf[k]) + bg_true[k]
        masks = [np.ones((m, n), bool), np.ones((m, n), bool)]
        model = _LinearModel3D(data, [obj1, obj2], psf, masks,
                               nominal_budget=1.0)
        sol = model.solve()
        np.testing.assert_allclose(sol[:2], amps_true, atol=1e-6)
        np.testing.assert_allclose(sol[2:], bg_true, atol=1e-6)
        pred = model.predict(sol)
        np.testing.assert_allclose(pred, data, atol=1e-4)

    def test_weights_gain_invariance(self):
        from sibilla.fit import _LinearModel3D
        from sibilla.forward import FastBlur

        nz, m, n = 3, 16, 16
        obj = np.zeros((m, n)); obj[7:9, 7:9] = 0.5
        yy, xx = np.mgrid[0:m, 0:n]
        g = np.exp(-((yy - 8) ** 2 + (xx - 8) ** 2) / 4.0)
        psf = np.stack([g / g.sum()] * nz)
        data = 100.0 * FastBlur(obj).blur(psf[0])[None] + 20.0
        data = np.concatenate([data] * nz)
        masks = [np.ones((m, n), bool)]
        m1 = _LinearModel3D(data, [obj], psf, masks, nominal_budget=1.0)
        s1 = m1.solve()
        m10 = _LinearModel3D(data * 10.0, [obj], psf, masks,
                             nominal_budget=1.0)
        s10 = m10.solve()
        # amplitudes and backgrounds scale by 10; normalized amps identical
        np.testing.assert_allclose(s10[:1] / s1[:1], [10.0], rtol=1e-6)
        np.testing.assert_allclose(s10[1:] / s1[1:], [10.0] * nz, rtol=1e-6)

    def test_per_plane_background_absorbs_slope(self):
        # Review Focus 4: sloped background must be absorbed, Zernikes unbiased
        # (reduced grid: fov 96, 3 segments, narrow z grid — the full-size
        # behavior is exercised by the pre-flight and end-to-end suites)
        from sibilla.scene import generate_scene
        from dataclasses import replace

        spec = replace(BACTERIA_SCENE, fov_px=96, n_segments=3, n_planes=9)
        scene = generate_scene(spec, seed=5)
        nz = spec.n_planes
        bg_slope = np.linspace(200.0, 600.0, nz)      # counts per plane
        data, psf, z_planes = _forward_planar(scene, {6: 0.10}, 4500.0,
                                              bg_per_plane=bg_slope, seed=11)
        cfg = real_pupil_config(spec.fov_px)
        skel = scene.segments[:, :4]                  # planar: (S, 4)

        from sibilla.fit import FitInput
        fit_in = FitInput(
            data=data, config=cfg, skeleton=skel,
            z_planes_um=z_planes,
            nominal_budget=4500.0 * (spec.pixel_size_um ** -2) * 0 + 4500.0,
            fluorescent_diameter_um=0.65,
            z_offset_grid_nm=(-100.0, 0.0, 100.0),
            n_multi_start=1, max_nfev=60,
        )
        res_pp = _fit_with_per_plane_bg(fit_in)
        err_pp = abs(res_pp.coeffs.get(6, 0.0) - 0.10)
        assert err_pp < 0.03


def _fit_with_per_plane_bg(fit_in: FitInput):
    """fit() with the 3D linear model (per-plane bg) monkeypatched in."""
    from sibilla import fit as fit_mod

    orig = fit_mod._LinearModel

    class _Adapter(orig):
        def __init__(self, data, objs, psf, masks, nominal_budget):
            from sibilla.fit import _LinearModel3D
            self._m3 = _LinearModel3D(data, objs, psf, masks,
                                      nominal_budget,
                                      per_plane_bg=True, poisson_weights=True)
            self.b = self._m3.b
            self.A = self._m3.A

        def solve(self):
            return self._m3.solve()

        def predict(self, sol):
            return self._m3.predict(sol)

        def roi(self, pred):
            return self._m3.roi(pred)

    fit_mod._LinearModel = _Adapter
    try:
        return fit(fit_in)
    finally:
        fit_mod._LinearModel = orig


# --- Task 6: FitInput3D + fit_3d --------------------------------------------


class TestFit3D:
    @staticmethod
    def _capsule_data(seed=3, coeffs=None, z_scatter=1.5, fov=96, n_seg=4,
                      peak=4500.0, bg=250.0):
        """Capsule-scene forward data, physically-correct depth model.

        Each rod is placed at a z-center scattered over `z_scatter` µm and
        blurred with the PSF evaluated at ITS OWN depth (cyclic-z 3D
        convolution) — the same model run_preflight uses. The returned
        skeleton encodes each rod's depth, so the fitter's per-segment
        z-candidate is a correction that should recover to ~0.
        """
        from sibilla.scene import CapsuleSpec, generate_capsule_scene
        from sibilla.forward import cyclic3d, psf3d_cyclic

        coeffs = coeffs or {}
        spec = CapsuleSpec(fov_px_y=fov, fov_px_x=fov, n_planes=15,
                           pixel_size_um=0.1135, z_step_um=0.5,
                           diameter_um=0.65, aa_width_px=1.0)
        scene = generate_capsule_scene(spec, seed=seed, n_segments=n_seg,
                                       length_range_um=(1.0, 3.0),
                                       z_center_range_um=(-z_scatter, z_scatter),
                                       max_tilt_deg=0.0)
        cfg = real_pupil_config(fov)
        z_planes = (np.arange(15) - 7) * 0.5
        psf = psf3d_cyclic(cfg, coeffs, 15, 0.5)
        rng = np.random.default_rng(seed + 100)
        data = np.full((15, fov, fov), bg)
        for obj in scene.objects:
            data += peak * cyclic3d(obj, psf) / obj.max()
        return rng.poisson(data).astype(np.float64), cfg, z_planes, scene, spec

    def test_fit3d_recovers_depth_scatter(self):
        # The point: the fitter recovers the shared Zernikes even when the
        # rods sit at scattered depths (the old crop-center PSF model failed
        # this). z-scatter ±1.0 µm = several planes, well beyond center.
        from sibilla.fit import FitInput3D, fit_3d

        data, cfg, z_planes, scene, spec = self._capsule_data(
            coeffs={6: 0.10}, z_scatter=1.0)
        fit_in = FitInput3D(
            data=data, config=cfg, skeleton=scene.segments,
            z_planes_um=z_planes,
            nominal_budget=4500.0,
            fluorescent_diameter_um=0.65,
            z_candidate_um=(-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5),
            n_multi_start=1, max_nfev=40,
        )
        res = fit_3d(fit_in)
        err6 = abs(res.coeffs.get(6, 0.0) - 0.10)
        assert err6 < 0.03
        # skeleton already carries each rod's depth -> correction ~ 0 (within
        # one candidate step, 0.5 µm)
        assert np.all(np.abs(res.segment_z_um) <= 0.5 + 1e-9)

    def test_fit3d_gain_invariance(self):
        # Review Focus 1: camera counts with unknown gain
        from sibilla.fit import FitInput3D, fit_3d

        data, cfg, z_planes, scene, spec = self._capsule_data(
            coeffs={5: 0.08})
        common = dict(
            config=cfg, skeleton=scene.segments, z_planes_um=z_planes,
            nominal_budget=4500.0, fluorescent_diameter_um=0.65,
            z_candidate_um=(-1.0, 0.0, 1.0), n_multi_start=1, max_nfev=30,
        )
        r1 = fit_3d(FitInput3D(data=data, **common))
        r10 = fit_3d(FitInput3D(data=data * 10.0, **common))
        for j in (5, 6, 7, 8):
            assert r1.coeffs.get(j, 0.0) == pytest.approx(
                r10.coeffs.get(j, 0.0), abs=1e-6)

    def test_fit3d_null(self):
        from sibilla.fit import FitInput3D, fit_3d

        data, cfg, z_planes, scene, spec = self._capsule_data(coeffs={})
        fit_in = FitInput3D(
            data=data, config=cfg, skeleton=scene.segments,
            z_planes_um=z_planes,
            nominal_budget=4500.0, fluorescent_diameter_um=0.65,
            z_candidate_um=(-1.0, 0.0, 1.0), n_multi_start=1, max_nfev=30,
        )
        res = fit_3d(fit_in)
        assert all(abs(v) < 0.01 for v in res.coeffs.values())

    def test_fit_from_real_files_smoke(self):
        from sibilla.fit import fit_from_real_files

        data, cfg, z_planes, scene, spec = self._capsule_data(
            coeffs={6: 0.05}, fov=96, n_seg=3)
        meta = {
            "pixel_size_um": 0.1135, "z_step_um": 0.5, "lambda_um": 0.517,
            "na": 1.2, "n_im": 1.33, "fluorescent_diameter_um": 0.65,
            "skeleton": scene.segments.tolist(),
            "z_planes_um": z_planes.tolist(),
            "nominal_budget": 4500.0,
        }
        res = fit_from_real_files(data, meta, max_nfev=20)
        assert isinstance(res.coeffs, dict)


class TestAlternatingOuterLoop:
    """Regression (2026-10-20): CD inside the residual made it discontinuous
    in c (z flips near ties) -> FD Jacobian garbage -> least_squares stalled
    at the start point and returned c=0 for nonzero injected aberration.
    Small fast scene reproduces it in ~1 min; must recover Z6."""

    def test_small_scene_recovers(self):
        import numpy as np
        from sibilla.defaults import real_pupil_config
        from sibilla.forward import cyclic3d, psf3d_cyclic
        from sibilla.scene import CapsuleSpec, generate_capsule_scene
        from sibilla.fit import FitInput3D, fit_3d

        fov, nz = 64, 9
        spec = CapsuleSpec(fov_px_y=fov, fov_px_x=fov, n_planes=nz,
                           pixel_size_um=0.1135, z_step_um=0.5,
                           diameter_um=0.65, aa_width_px=1.0)
        scene = generate_capsule_scene(spec, seed=3, n_segments=4,
                                       length_range_um=(1.0, 1.5),
                                       z_center_range_um=(-0.5, 0.5),
                                       max_tilt_deg=0.0)
        cfg = real_pupil_config(fov)
        psf = psf3d_cyclic(cfg, {6: 0.05}, nz, 0.5)
        rng = np.random.default_rng(7)
        data = np.full((nz, fov, fov), 250.0)
        for obj in scene.objects:
            data += 4500.0 * cyclic3d(obj, psf) / obj.max()
        data = rng.poisson(data).astype(np.float64)
        z_planes = (np.arange(nz) - (nz - 1) / 2) * 0.5
        fit_in = FitInput3D(data=data, config=cfg, skeleton=scene.segments,
                            z_planes_um=z_planes, nominal_budget=4500.0,
                            fluorescent_diameter_um=0.65,
                            z_candidate_um=(-1.0, -0.5, 0.0, 0.5, 1.0),
                            n_multi_start=0, max_nfev=25)
        res = fit_3d(fit_in)
        assert abs(res.coeffs.get(6, 0.0) - 0.05) < 0.02, res.coeffs

    def test_optimizer_actually_moves(self):
        # The guard whose absence let "optimizer never runs" hide for a day:
        # final cost must be meaningfully BELOW the cost at the start point
        # (c=0, z=mid). If least_squares dies instantly (e.g. an exception
        # inside residuals), the fit "succeeds" at the start with c=0 and
        # this catches it.
        import numpy as np
        from sibilla.defaults import real_pupil_config
        from sibilla.forward import cyclic3d, psf3d_cyclic
        from sibilla.scene import CapsuleSpec, generate_capsule_scene
        from sibilla.fit import FitInput3D, fit_3d

        fov, nz = 64, 9
        spec = CapsuleSpec(fov_px_y=fov, fov_px_x=fov, n_planes=nz,
                           pixel_size_um=0.1135, z_step_um=0.5,
                           diameter_um=0.65, aa_width_px=1.0)
        scene = generate_capsule_scene(spec, seed=3, n_segments=4,
                                       length_range_um=(1.0, 1.5),
                                       z_center_range_um=(-0.5, 0.5),
                                       max_tilt_deg=0.0)
        cfg = real_pupil_config(fov)
        psf = psf3d_cyclic(cfg, {6: 0.05}, nz, 0.5)
        rng = np.random.default_rng(7)
        data = np.full((nz, fov, fov), 250.0)
        for obj in scene.objects:
            data += 4500.0 * cyclic3d(obj, psf) / obj.max()
        data = rng.poisson(data).astype(np.float64)
        z_planes = (np.arange(nz) - (nz - 1) / 2) * 0.5
        fit_in = FitInput3D(data=data, config=cfg, skeleton=scene.segments,
                            z_planes_um=z_planes, nominal_budget=4500.0,
                            fluorescent_diameter_um=0.65,
                            z_candidate_um=(-1.0, -0.5, 0.0, 0.5, 1.0),
                            n_multi_start=0, max_nfev=25)
        res = fit_3d(fit_in)
        # fit_3d's own residuals carry the full-image residual at c; the
        # reported cost must be clearly below the c=0 baseline we measured
        # on this scene (~1.9e5 with the broken version; healthy fit lands
        # ~1e4). Assert an order-of-magnitude drop.
        assert res.coeffs.get(6, 0.0) != 0.0, (
            "optimizer returned the start point — least_squares never ran "
            f"or never moved (coeffs={res.coeffs})")


class TestCoeffBound:
    """2026-11-03 field-test follow-up: real-data fits pinned at the
    hardcoded ±0.5 bound. The bound is now a FitInput3D field
    (`coeff_bound`, default 0.5 — so all existing tests keep pinning the
    physics conventions). This guard: a 0.7-wave aberration must PIN at the
    default ±0.5 (lower bound only) but RECOVER when the bound is widened."""

    @staticmethod
    def _big_aberration_data(seed=3):
        from sibilla.forward import cyclic3d, psf3d_cyclic
        from sibilla.scene import CapsuleSpec, generate_capsule_scene

        fov, nz = 64, 9
        spec = CapsuleSpec(fov_px_y=fov, fov_px_x=fov, n_planes=nz,
                           pixel_size_um=0.1135, z_step_um=0.5,
                           diameter_um=0.65, aa_width_px=1.0)
        scene = generate_capsule_scene(spec, seed=seed, n_segments=4,
                                       length_range_um=(1.0, 1.5),
                                       z_center_range_um=(-0.5, 0.5),
                                       max_tilt_deg=0.0)
        cfg = real_pupil_config(fov)
        psf = psf3d_cyclic(cfg, {6: 0.7}, nz, 0.5)
        rng = np.random.default_rng(seed + 7)
        data = np.full((nz, fov, fov), 250.0)
        for obj in scene.objects:
            data += 4500.0 * cyclic3d(obj, psf) / obj.max()
        data = rng.poisson(data).astype(np.float64)
        z_planes = (np.arange(nz) - (nz - 1) / 2) * 0.5
        return data, cfg, scene, z_planes

    def test_default_bound_pins(self):
        from sibilla.fit import FitInput3D, fit_3d

        data, cfg, scene, z_planes = self._big_aberration_data()
        fit_in = FitInput3D(data=data, config=cfg, skeleton=scene.segments,
                            z_planes_um=z_planes, nominal_budget=4500.0,
                            fluorescent_diameter_um=0.65,
                            z_candidate_um=(-1.0, -0.5, 0.0, 0.5, 1.0),
                            n_multi_start=0, max_nfev=60)
        res = fit_3d(fit_in)
        z6 = res.coeffs.get(6, 0.0)
        # data wants 0.7 -> optimizer pushes to the default ±0.5 wall
        assert abs(z6) >= 0.5 - 1e-9, res.coeffs

    def test_wider_bound_recovers(self):
        from sibilla.fit import FitInput3D, fit_3d

        data, cfg, scene, z_planes = self._big_aberration_data()
        fit_in = FitInput3D(data=data, config=cfg, skeleton=scene.segments,
                            z_planes_um=z_planes, nominal_budget=4500.0,
                            fluorescent_diameter_um=0.65,
                            z_candidate_um=(-1.0, -0.5, 0.0, 0.5, 1.0),
                            coeff_bound=1.5,
                            n_multi_start=0, max_nfev=120)
        res = fit_3d(fit_in)
        z6 = res.coeffs.get(6, 0.0)
        assert abs(z6 - 0.7) < 0.03, res.coeffs
        assert abs(z6) < 1.5 - 1e-9, "still pinned at the wider bound"
