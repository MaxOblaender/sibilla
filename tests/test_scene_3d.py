"""Tests for the 3D capsule object model (Task 4)."""

from __future__ import annotations

import numpy as np
import pytest

from sibilla.scene import (
    CapsuleScene,
    CapsuleSpec,
    generate_capsule_scene,
    render_capsule_unit,
)


def _spec(n_planes: int = 15, d: float = 0.65) -> CapsuleSpec:
    return CapsuleSpec(
        fov_px_y=96, fov_px_x=96, n_planes=n_planes,
        pixel_size_um=0.1135, z_step_um=0.5,
        diameter_um=d, aa_width_px=1.0,
    )


class TestRenderCapsuleUnit:
    def test_unit_sum(self):
        spec = _spec()
        obj = render_capsule_unit(20, 40, 7, 70, 50, 7, spec)
        assert obj.shape == (spec.n_planes, 96, 96)
        assert obj.sum() == pytest.approx(1.0, abs=1e-12)

    def test_untilted_footprint_area(self):
        # z0 == z1, single plane: swept-disk area ≈ π r²
        spec = _spec(n_planes=1)
        r_um = spec.diameter_um / 2
        obj = render_capsule_unit(30, 48, 0, 60, 48, 0, spec)
        px_area = spec.pixel_size_um ** 2
        # swept capsule: rectangle L×2r + full disk (two half-caps) + AA ring
        L_um = 30 * spec.pixel_size_um
        expected_n = (L_um * 2 * r_um + np.pi * r_um ** 2) / px_area
        ring = (2 * L_um + 2 * np.pi * r_um) * spec.pixel_size_um / px_area
        n_cov = (obj > 0).sum()
        assert n_cov == pytest.approx(expected_n + ring, rel=0.15)

    def test_tilted_centroid_follows_axis(self):
        # 30°-tilted capsule: centroid x on plane k interpolates the axis
        spec = _spec()
        x0, x1 = 30.0, 60.0
        z0, z1 = 3.0, 11.0
        obj = render_capsule_unit(x0, 48, z0, x1, 48, z1, spec)
        for k in (4, 7, 10):
            t = (k - z0) / (z1 - z0)
            cx_expect = x0 + t * (x1 - x0)
            prof = obj[k].sum(axis=0)
            cx = (prof * np.arange(96)).sum() / prof.sum()
            assert cx == pytest.approx(cx_expect, abs=1.5)

    def test_untilted_z_profile_unimodal(self):
        spec = _spec()
        obj = render_capsule_unit(40, 48, 7, 56, 48, 7, spec)
        zp = obj.sum(axis=(1, 2))
        peak = int(np.argmax(zp))
        rising = np.all(np.diff(zp[:peak + 1]) >= -1e-12)
        falling = np.all(np.diff(zp[peak:]) <= 1e-12)
        assert rising and falling


class TestGenerateCapsuleScene:
    def test_scene_shape_and_constraints(self):
        spec = _spec()
        scene = generate_capsule_scene(spec, seed=0, n_segments=5,
                                       length_range_um=(1.0, 3.0),
                                       z_center_range_um=(-1.0, 1.0),
                                       max_tilt_deg=30.0)
        assert isinstance(scene, CapsuleScene)
        assert scene.segments.shape == (5, 6)
        assert len(scene.angles_rad) == 5
        assert len(scene.objects) == 5
        for o in scene.objects:
            assert o.sum() == pytest.approx(1.0, abs=1e-12)
            assert np.isfinite(o).all()
        assert np.all(scene.tilts_deg <= 30.0 + 1e-9)

    def test_reproducible(self):
        spec = _spec()
        a = generate_capsule_scene(spec, seed=3, n_segments=4)
        b = generate_capsule_scene(spec, seed=3, n_segments=4)
        np.testing.assert_allclose(a.segments, b.segments)
