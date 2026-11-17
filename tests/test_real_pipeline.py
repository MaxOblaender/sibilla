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
        """Ensemble over two INDEPENDENT valid crops sharing the aberration.

        Each crop is a self-consistent fit on its own (rods inside with margin,
        correct skeleton) — the real use case. The old test sliced ONE volume
        into two 64² halves; each half was an invalid fragment (rods cut by the
        edge, skeleton listing rods that weren't there), which is what produced
        the pinned, wrong-crop result (plans/fix-ensemble-pinned-crops.md).
        """
        from sibilla.real_pipeline import fit_ensemble

        coeffs = {5: 0.08, 6: -0.06}
        # two independent crops of the same field: same PSF/aberration,
        # different rod layouts (different seeds), each a valid fit
        crop_metas = []
        for name, seed in {"A": 3, "B": 4}.items():
            vol, skel, offs, z_planes = _volume_with_rods(
                n_seg=3, coeffs=coeffs, fov=96, seed=seed)
            crop_metas.append({
                "name": name, "pixel_size_um": 0.1135, "z_step_um": 0.5,
                "lambda_um": 0.517, "na": 1.2, "n_im": 1.33,
                "fluorescent_diameter_um": 0.65, "skeleton": skel.tolist(),
                "n_segments": len(skel),
                "z_planes_um": z_planes.tolist(), "nominal_budget": 4500.0,
                "_data": vol,
            })
        out_dir = Path("results/_test_ensemble")
        out_dir.mkdir(parents=True, exist_ok=True)
        res = fit_ensemble(crop_metas, out_dir)
        assert "coeffs" in res
        # NOTE: the report serializes mode numbers as STRING keys ("5", "6", …)
        # — the old test read them with int keys and silently got 0.0 back,
        # which is what made the failure look like a fit stall.
        err5 = abs(res["coeffs"].get("5", 0.0) - 0.08)
        err6 = abs(res["coeffs"].get("6", 0.0) - (-0.06))
        assert err5 < 0.04 and err6 < 0.04
        assert res["n_crops"] == 2
        assert res["n_crops_used"] == 2
        assert res["n_crops_rejected"] == 0
        assert (out_dir / "fit_report.json").exists()

    def test_ensemble_wiring_excludes_pinned_crop(self, monkeypatch, tmp_path):
        """fit_ensemble WIRING (no imaging): with one admissible and one
        bound-pinned crop, the report must exclude the pinned crop, keep the
        good one's coefficients, and record both names. Closes the gap found
        by adversarial review 2026-11-17 (policy was only tested directly).
        """
        from sibilla import real_pipeline
        from sibilla.fit import FitResult

        good = {5: 0.08, 6: -0.06, 7: 0.004, 8: -0.002}
        pinned = {5: -0.5, 6: 0.5, 7: -0.476, 8: 0.270}

        def fake_fit_3d(fit_in):
            return FitResult(coeffs=dict(_next.pop(0)),
                             amplitudes=np.zeros(1), background=0.0,
                             z_offset_um=0.0, cost=1.0,
                             residuals=np.zeros((1, 1, 1)))

        # call order: main loop over crops (good, pinned), then λ refits for
        # ADMISSIBLE crops only (good, good). A StopIteration here means the
        # call pattern changed — update this test deliberately, not silently.
        _next = [good, pinned, good, good]
        monkeypatch.setattr(real_pipeline, "fit_3d", fake_fit_3d)
        metas = []
        for i, name in enumerate(("crop0", "crop1")):
            metas.append({
                "name": name, "pixel_size_um": 0.1135, "z_step_um": 0.5,
                "lambda_um": 0.517, "na": 1.2, "n_im": 1.33,
                "fluorescent_diameter_um": 0.65,
                "skeleton": [[10 + i, 10, 7, 20 + i, 10, 7]],
                "n_segments": 1, "z_planes_um": [(k - 7) * 0.5 for k in range(15)],
                "nominal_budget": 4500.0, "_data": np.zeros((15, 16, 16)),
            })
        res = real_pipeline.fit_ensemble(metas, tmp_path)
        assert res["n_crops_used"] == 1
        assert res["n_crops_rejected"] == 1
        assert res["used_crops"] == ["crop0"]
        assert res["rejected_crops"] == ["crop1"]
        assert abs(res["coeffs"]["5"] - good[5]) < 1e-12
        assert abs(res["coeffs"]["6"] - good[6]) < 1e-12

    def test_ensemble_wiring_all_pinned_raises(self, monkeypatch, tmp_path):
        """All crops pinned -> fit_ensemble must propagate the loud failure
        (the old code silently averaged pinned crops into a garbage estimate)."""
        from sibilla import real_pipeline
        from sibilla.fit import FitResult

        pinned = {5: -0.5, 6: 0.5, 7: -0.476, 8: 0.270}
        monkeypatch.setattr(real_pipeline, "fit_3d", lambda fit_in: FitResult(
            coeffs=dict(pinned), amplitudes=np.zeros(1), background=0.0,
            z_offset_um=0.0, cost=1.0, residuals=np.zeros((1, 1, 1))))
        metas = [{"name": f"crop{i}", "pixel_size_um": 0.1135,
                  "z_step_um": 0.5, "lambda_um": 0.517, "na": 1.2, "n_im": 1.33,
                  "fluorescent_diameter_um": 0.65, "skeleton": [[10, 10, 7, 20, 10, 7]],
                  "n_segments": 1, "z_planes_um": [(k - 7) * 0.5 for k in range(15)],
                  "nominal_budget": 4500.0, "_data": np.zeros((15, 16, 16))}
                 for i in range(2)]
        with pytest.raises(ValueError, match="pinned"):
            real_pipeline.fit_ensemble(metas, tmp_path)

