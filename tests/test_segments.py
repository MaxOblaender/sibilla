"""Tests for segments.py (Task 7)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

skimage = pytest.importorskip("skimage")

from sibilla.segments import (  # noqa: E402
    SegmentsResult,
    extract_segments,
    make_review_figure,
    read_segments_csv,
    write_segments_csv,
)
from sibilla.scene import (  # noqa: E402
    CapsuleSpec,
    generate_capsule_scene,
    render_capsule_unit,
)


def _spec(fov=96, n_planes=15) -> CapsuleSpec:
    return CapsuleSpec(fov_px_y=fov, fov_px_x=fov, n_planes=n_planes,
                       pixel_size_um=0.1135, z_step_um=0.5,
                       diameter_um=0.65, aa_width_px=1.0)


def _render_scene_to_stack(scene, peak_per_voxel=4500.0, bg=250.0, seed=0):
    """Render capsule objects with per-VOXEL peak brightness ~4500 counts.

    (The real crop's rods peak ~4500 counts/voxel over ~270 background —
    spec §1. Unit-sum capsule footprints are rescaled accordingly.)
    """
    nz = scene.spec.n_planes
    m = scene.spec.fov_px_y
    n = scene.spec.fov_px_x
    rng = np.random.default_rng(seed)
    data = np.full((nz, m, n), bg)
    for obj in scene.objects:
        data += peak_per_voxel * obj / obj.max()
    return rng.poisson(data).astype(np.float32)


class TestExtractSegments:
    def test_recovers_synthetic_rods(self):
        spec = _spec()
        scene = generate_capsule_scene(spec, seed=7, n_segments=5,
                                       length_range_um=(1.5, 2.5),
                                       z_center_range_um=(-1.0, 1.0),
                                       max_tilt_deg=0.0)
        stack = _render_scene_to_stack(scene)
        res = extract_segments(stack, pixel_um=0.1135, z_step_um=0.5)
        assert isinstance(res, SegmentsResult)
        assert res.segments.shape[1] == 6
        assert len(res.segments) >= 4
        assert np.isfinite(res.segments).all()
        # angles in [0, 180)
        for seg in res.segments:
            dx, dy = seg[3] - seg[0], seg[4] - seg[1]
            ang = np.degrees(np.arctan2(dy, dx)) % 180.0
            assert 0.0 <= ang < 180.0
        # most rods unflagged
        n_flagged = sum(1 for fl in res.flags.values() if fl)
        assert n_flagged == 0
        # endpoints near a true rod's endpoints (3 px, best matching)
        for seg in res.segments:
            d = np.abs(scene.segments - seg[None, :]).sum(axis=1)
            assert d.min() <= 3.0 * 6, scene.segments[d.argmin()]

    def test_sheared_rod_flagged(self):
        # Review Focus 3: a rod split into two z-disjoint halves
        spec = _spec()
        stack = np.full((15, 96, 96), 250.0, np.float32)
        rng = np.random.default_rng(1)
        # two half-rods, same xy line, z bands [1,3] and [8,10]
        a = render_capsule_unit(30, 40, 1, 60, 52, 3, spec)
        b = render_capsule_unit(30, 40, 8, 60, 52, 10, spec)
        stack += 4500.0 * (a + b)
        stack = rng.poisson(stack).astype(np.float32)
        res = extract_segments(stack, pixel_um=0.1135, z_step_um=0.5)
        assert any("sheared" in fl for fl in res.flags.values())

    def test_saturated_flagged(self):
        # Review Focus 2: a saturated rod (16383 plateau)
        spec = _spec()
        scene = generate_capsule_scene(spec, seed=2, n_segments=2,
                                       length_range_um=(1.5, 2.5),
                                       z_center_range_um=(-0.5, 0.5),
                                       max_tilt_deg=0.0)
        stack = _render_scene_to_stack(scene)
        stack[:, 44:50, 30:36] = 16383.0
        res = extract_segments(stack, pixel_um=0.1135, z_step_um=0.5)
        assert any("saturated" in fl for fl in res.flags.values())


class TestCsvRoundTrip:
    def test_round_trip(self, tmp_path: Path):
        spec = _spec()
        scene = generate_capsule_scene(spec, seed=7, n_segments=4,
                                       length_range_um=(1.5, 2.5),
                                       z_center_range_um=(-1.0, 1.0),
                                       max_tilt_deg=0.0)
        stack = _render_scene_to_stack(scene)
        res = extract_segments(stack, pixel_um=0.1135, z_step_um=0.5)
        csv = tmp_path / "segments.csv"
        write_segments_csv(csv, res)
        back = read_segments_csv(csv)
        # CSV stores 3 decimals (review format) — match that precision
        np.testing.assert_allclose(back.segments, res.segments, atol=1e-3)
        assert back.flags == res.flags

    def test_make_review_figure(self, tmp_path: Path):
        spec = _spec()
        scene = generate_capsule_scene(spec, seed=7, n_segments=3,
                                       length_range_um=(1.5, 2.5),
                                       z_center_range_um=(-1.0, 1.0),
                                       max_tilt_deg=0.0)
        stack = _render_scene_to_stack(scene)
        res = extract_segments(stack, pixel_um=0.1135, z_step_um=0.5)
        png = tmp_path / "review.png"
        make_review_figure(stack, res, png)
        assert png.exists() and png.stat().st_size > 10_000


class TestRealCrop:
    def test_delivered_crop_segments(self):
        crop = Path(__file__).resolve().parents[2] / "GoC-Exp4_C3_7-1_3d.tif"
        if not crop.exists():
            pytest.skip("3D crop not present")
        import tifffile

        stack = tifffile.imread(crop).astype(np.float32)
        stack = np.pad(stack, ((0, 0), (0, 72), (0, 0)),
                       mode="constant",
                       constant_values=float(np.median(stack)))
        res = extract_segments(stack, pixel_um=0.1135, z_step_um=0.5)
        # 2026-10-19 reality: this crop has ~4 clean single rods; the rest
        # are xy-colinear chains of touching bacteria at different depths
        # (flagged 'sheared' by the z-drift rule — correctly). The ensemble
        # target (>= 20 rods) is met across MULTIPLE crops (spec §4).
        assert 6 <= len(res.segments) <= 30
        n_unflagged = sum(1 for fl in res.flags.values() if not fl)
        assert n_unflagged >= 4
