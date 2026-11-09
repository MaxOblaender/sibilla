import numpy as np
import pytest
from sibilla.twod import flatfield

def _texture(seed=0, n=128):
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(n, n))
    return np.clip(0.5 + 0.5 * base, 0, 1).astype(np.uint8) * 200

def test_l1_fluorescence_removes_level_and_scale():
    tile = _texture()
    out, bg, scale = flatfield.l1_normalize(tile, "fluorescence")
    assert out.dtype == np.float32
    assert bg == pytest.approx(np.median(tile[tile <= np.median(tile)]), abs=1.0)
    assert np.median(out) == pytest.approx(0.0, abs=0.02)

def test_l1_transmission_inverts_darkness():
    bg_level = np.full((128, 128), 240, np.uint8)
    bg_level[40:80, 40:80] = 20
    out, bg, scale = flatfield.l1_normalize(bg_level, "transmission")
    assert bg == pytest.approx(240.0, abs=2.0)
    assert out[60, 60] > out[5, 5] * 5   # tissue (dark) is bright after inversion

def test_l1_zero_scale_raises():
    with pytest.raises(ValueError):
        flatfield.l1_normalize(np.zeros((32, 32), np.uint8), "fluorescence")

def test_l2_recovers_known_gain_and_offset():
    ref = _texture(1).astype(np.float32) / 255.0
    target = (0.6 * ref + 0.2).astype(np.float32)
    matched, g, c = flatfield.l2_match(ref, target, f_max=0.15)
    assert g == pytest.approx(0.6, rel=0.05)
    assert c == pytest.approx(0.2, abs=0.02)
    np.testing.assert_allclose(matched, ref, atol=0.03)

def test_l2_is_identity_for_identical_patches():
    ref = _texture(2).astype(np.float32) / 255.0
    matched, g, c = flatfield.l2_match(ref, ref, f_max=0.15)
    assert g == pytest.approx(1.0, rel=0.02)
    assert c == pytest.approx(0.0, abs=0.01)

def test_l2_degenerate_pair_returns_target_uncorrected():
    rng = np.random.default_rng(9)
    ref = _texture(3).astype(np.float32) / 255.0
    target = rng.random((128, 128)).astype(np.float32)   # uncorrelated -> g ~ 0
    matched, g, c = flatfield.l2_match(ref, target, f_max=0.15)
    assert abs(g) < 0.1
    np.testing.assert_allclose(matched, target)

def test_estimate_empty_background():
    tile = np.full((64, 64), 12.0, np.float32)
    tile += np.random.default_rng(7).normal(scale=0.5, size=tile.shape).astype(np.float32)
    tile[:8, :8] = 250.0               # small bright object
    med, sig = flatfield.estimate_empty_background(tile)
    assert med == pytest.approx(12.0, abs=1.0)
    assert 0.0 < sig < 3.0
