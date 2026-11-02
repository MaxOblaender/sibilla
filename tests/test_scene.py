"""Unit tests for the tubule-network scene generator (plan §2)."""

import numpy as np
import pytest

from sibilla.defaults import DEFAULT_SCENE, SceneSpec
from sibilla.scene import (
    _max_angular_gap_deg,
    generate_scene,
    render_tubule_unit,
)


def test_generate_reaches_segment_count():
    scene = generate_scene(DEFAULT_SCENE, seed=0)
    assert scene.n_segments == DEFAULT_SCENE.n_segments


def test_segment_lengths_in_range():
    scene = generate_scene(DEFAULT_SCENE, seed=1)
    lengths = scene.length_um
    assert np.all(lengths >= DEFAULT_SCENE.length_range_um[0] - 1e-6)
    assert np.all(lengths <= DEFAULT_SCENE.length_range_um[1] + 1e-6)


def test_min_separation_respected():
    scene = generate_scene(DEFAULT_SCENE, seed=2)
    segs = scene.segments
    # check all pairwise distances >= ~min_sep (use the same metric the
    # generator uses, minus a tiny tolerance for float error)
    from sibilla.scene import _segments_min_dist_um
    for i in range(len(segs)):
        for j in range(i + 1, len(segs)):
            d = _segments_min_dist_um(segs[i], segs[j], DEFAULT_SCENE)
            assert d >= 0.79, f"segments {i},{j} too close: {d}"


def test_orientation_coverage():
    scene = generate_scene(DEFAULT_SCENE, seed=3)
    gap = _max_angular_gap_deg(scene.angles_rad)
    assert gap <= DEFAULT_SCENE.max_angular_gap_deg + 1e-6


def test_border_margin():
    scene = generate_scene(DEFAULT_SCENE, seed=4)
    margin = DEFAULT_SCENE.border_margin_um / DEFAULT_SCENE.pixel_size_um
    for seg in scene.segments:
        assert seg.min() >= margin - 1e-6
        assert seg.max() <= DEFAULT_SCENE.fov_px - margin + 1e-6


def test_render_unit_photon_integral():
    img = render_tubule_unit(10.0, 20.0, 60.0, 50.0, DEFAULT_SCENE)
    assert img.shape == (DEFAULT_SCENE.fov_px, DEFAULT_SCENE.fov_px)
    # unit photons: integral exactly 1
    assert img.sum() == pytest.approx(1.0, rel=1e-12)
    assert np.all(img >= 0)


def test_render_diameter_scaling():
    """A larger fluorescent diameter yields a wider (lower-peak) profile."""
    img_small = render_tubule_unit(30.0, 50.0, 90.0, 50.0, DEFAULT_SCENE,
                                   diameter_um=0.060)
    img_big = render_tubule_unit(30.0, 50.0, 90.0, 50.0, DEFAULT_SCENE,
                                 diameter_um=0.120)
    # both unit integral
    assert img_small.sum() == pytest.approx(1.0)
    assert img_big.sum() == pytest.approx(1.0)
    # the big one spreads energy -> lower peak
    assert img_big.max() < img_small.max()


def test_max_angular_gap_known():
    # all parallel segments -> one cluster, gap = 180 deg (mod pi coverage)
    assert _max_angular_gap_deg(np.array([0.0, 0.0])) == pytest.approx(180.0, abs=1e-9)
    # perpendicular pair -> gaps of 90 deg each
    assert _max_angular_gap_deg(np.array([0.0, np.pi / 2])) == pytest.approx(90.0, abs=1e-6)
    # three evenly spread (0, 60, 120) -> gaps of 60 deg each
    a = np.array([0.0, np.pi / 3, 2 * np.pi / 3])
    assert _max_angular_gap_deg(a) == pytest.approx(60.0, abs=1e-6)
