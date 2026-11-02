"""Tests for real_pipeline.py (Task 10)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from sibilla.scene import CapsuleSpec, generate_capsule_scene
from sibilla.run_preflight import make_capsule_data, _capsule_spec
from sibilla.defaults import real_pupil_config


def _volume_with_rods(n_seg=5, coeffs=None, fov=128, z_scatter=1.0,
                      peak=4500.0, bg=250.0, seed=0):
    """A full (Nz, M, N) volume with scattered rods + its skeleton.

    Physically-correct depth model (cyclic-z 3D convolution): each rod is
    blurred with the PSF evaluated at its OWN depth (its scattered z-center).
    Returns (data, skeleton, per-rod z-offsets µm, symmetric z_planes_um).
    """
    from sibilla.forward import cyclic3d, psf3d_cyclic

    spec = _capsule_spec(fov, n_planes=15)
    rng = np.random.default_rng(seed)
    cfg = real_pupil_config(fov)
    z_planes = (np.arange(15) - 7) * 0.5
    scene = generate_capsule_scene(spec, seed=seed, n_segments=n_seg,
                                   length_range_um=(1.5, 3.0),
                                   z_center_range_um=(-z_scatter, z_scatter),
                                   max_tilt_deg=0.0)
    psf = psf3d_cyclic(cfg, coeffs or {}, 15, 0.5)
    data = np.full((15, fov, fov), bg)
    offs = []
    for seg, obj in zip(scene.segments, scene.objects):
        offs.append(float(np.mean(seg[[2, 5]]) * 0.5))  # µm around center
        data += peak * cyclic3d(obj, psf) / max(float(obj.max()), 1e-12)
    rng2 = np.random.default_rng(seed + 7)
    return rng2.poisson(data).astype(np.float64), scene.segments, offs, z_planes


class TestAutoCrops:
    def test_finds_planted_rods(self):
        from sibilla.real_pipeline import auto_crops

        vol, skel, offs, _ = _volume_with_rods(n_seg=5, seed=3)
        crops = auto_crops(vol, 0.1135, max_crops=8, crop_hw=24, n_planes=15)
        assert len(crops) >= 4
        # each crop center near a true rod center
        for (z0, y0, x0) in crops:
            cy = y0 + 24
            cx = x0 + 24
            cz = z0 + 7
            d = np.hypot(skel[:, 1] - cy, skel[:, 0] - cx)
            assert d.min() <= 40, (crops, skel)

    def test_veto_saturated(self):
        # Review Focus 2: a saturated rod is NOT cropped
        from sibilla.real_pipeline import auto_crops

        vol, skel, offs, _ = _volume_with_rods(n_seg=2, seed=4)
        # saturate one rod's location
        seg = skel[0]
        for k in range(5, 10):
            x0, y0 = int(seg[0]), int(seg[1])
            vol[k, max(0, y0 - 3):y0 + 4, max(0, x0 - 3):x0 + 4] = 16383.0
        crops = auto_crops(vol, 0.1135, max_crops=8, crop_hw=24, n_planes=15,
                           sat_level=16300)
        # the saturated rod's center should have no crop
        cy, cx = int(seg[1]), int(seg[0])
        for (z0, y0, x0) in crops:
            assert not (abs(y0 + 24 - cy) <= 20 and abs(x0 + 24 - cx) <= 20), \
                "saturated rod should be vetoed"


class TestG3Checks:
    def test_huge_coeff_fails(self):
        from sibilla.fit import FitResult
        from sibilla.real_pipeline import g3_checks

        data = np.zeros((3, 16, 16))
        res = FitResult(coeffs={6: 0.7}, amplitudes=np.array([1.0]),
                        background=0.0, z_offset_um=0.0, cost=1.0,
                        residuals=data)
        out = g3_checks(res, bootstrap={"sigma": {6: 0.01}})
        assert out["pass"] is False
        assert any("0.5" in i or "wave" in i.lower() for i in out["issues"])

    def test_structured_residuals_flagged(self):
        from sibilla.fit import FitResult
        from sibilla.real_pipeline import g3_checks

        nz = 6
        data = np.zeros((nz, 16, 16))
        # plane-mean residual trending (systematic structure)
        resid = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])[:, None, None]
        res = FitResult(coeffs={5: 0.02}, amplitudes=np.array([1.0]),
                        background=0.0, z_offset_um=0.0, cost=1.0,
                        residuals=resid)
        out = g3_checks(res, bootstrap={"sigma": {5: 0.01}})
        assert any("residual" in i.lower() or "structure" in i.lower()
                   for i in out["issues"])


class TestFitEnsemble:
    def test_synthetic_two_crop_ensemble(self):
        from sibilla.real_pipeline import fit_ensemble

        # one volume, cut into 2 crops that share the same PSF
        vol, skel, offs, z_planes = _volume_with_rods(
            n_seg=4, coeffs={5: 0.08, 6: -0.06}, fov=128, seed=5)
        half = 64
        crop_metas = []
        for name, (ys, xs) in {"A": (0, 0), "B": (64, 0)}.items():
            stack = vol[:, ys:ys + half, xs:xs + half].astype(np.float32)
            # re-locate skeleton into crop coordinates
            seg = skel.copy()
            seg[:, 1] -= ys
            seg[:, 3] -= ys
            seg[:, 0] -= xs
            seg[:, 2] -= xs
            crop_metas.append({
                "name": name, "pixel_size_um": 0.1135, "z_step_um": 0.5,
                "lambda_um": 0.517, "na": 1.2, "n_im": 1.33,
                "fluorescent_diameter_um": 0.65, "skeleton": seg.tolist(),
                "z_planes_um": z_planes.tolist(), "nominal_budget": 4500.0,
                "_data": stack,
            })
        out_dir = Path("results/_test_ensemble")
        out_dir.mkdir(parents=True, exist_ok=True)
        res = fit_ensemble(crop_metas, out_dir)
        assert "coeffs" in res
        err5 = abs(res["coeffs"].get(5, 0.0) - 0.08)
        err6 = abs(res["coeffs"].get(6, 0.0) - (-0.06))
        assert err5 < 0.04 and err6 < 0.04
        assert (out_dir / "fit_report.json").exists()
