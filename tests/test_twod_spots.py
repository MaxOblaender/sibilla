import numpy as np
import pytest
from sibilla.twod import spots

def _img_with_spots():
    img = np.full((256, 256), 0.05, np.float32)
    img += np.random.default_rng(0).normal(scale=0.005, size=img.shape).astype(np.float32)
    for (y, x) in [(64, 64), (128, 190), (200, 40)]:
        img[y - 1 : y + 2, x - 1 : x + 2] += 0.8      # PSF-sized point
    img[40:90, 160:200] = 0.9                          # colony-like blob
    img[220:230, 100:220] = 0.9                        # hypha-like bar
    return img

def test_detects_point_like_spots_only():
    res = spots.detect(_img_with_spots())
    acc = [s for s in res if s.accepted]
    assert len(acc) == 3
    assert {(round(s.x), round(s.y)) for s in acc} == {(64, 64), (190, 128), (40, 200)}

def test_colony_and_bar_are_rejected_with_reason():
    res = spots.detect(_img_with_spots())
    rej = [s for s in res if not s.accepted]
    reasons = {s.reason for s in rej}
    assert "too_large" in reasons or "elongated" in reasons

def test_flat_image_detects_nothing():
    assert spots.detect(np.full((128, 128), 0.05, np.float32)) == []

def test_sanity_visuals_written(tmp_path):
    img = _img_with_spots()
    res = spots.detect(img)
    paths = spots.write_sanity_visuals(img, res, tmp_path, spots.DetectionParams(), "P00001")
    names = {p.name for p in paths}
    assert "spots_tile_P00001.png" in names
    assert "spots_sheet.png" in names
    assert "spots_summary.png" in names
    csv = tmp_path / "spots_summary.csv"
    assert csv.exists()
    text = csv.read_text()
    assert "P00001" in text
    for col in ("x", "y", "score", "radius_px", "aspect", "accepted", "reason"):
        assert col in text.splitlines()[0]
    # rejected candidates must be in the record, not silently dropped
    assert text.count("False") >= 1 or "rejected" in text

def test_manual_override_takes_priority(tmp_path):
    p = tmp_path / "manual.json"
    p.write_text('[[10.0, 20.0], [30.0, 40.0]]')
    assert spots.load_manual(p) == [(10.0, 20.0), (30.0, 40.0)]

def test_thin_bar_is_rejected_as_elongated():
    """Pin for the elongated gate (final-review N1): a thin bar is far below
    the size gate (2x24 bar: equivalent radius ~3.9 px <= 8, second-moment
    aspect ~13.8) and must be rejected as elongated, in BOTH orientations.
    This is the gate whose aspect formula once slipped through inverted and
    left the whole suite green. The bar is 2 px wide ON PURPOSE: a perfectly
    straight 1-px line has lmin == 0 and hits the degenerate ``inf`` branch
    BEFORE the aspect ratio is evaluated, so it cannot catch a flipped
    formula (measured). The background carries noise: a perfectly flat one
    has sigma_bg == 0 and detect() refuses to threshold (returns [])."""
    img = np.full((128, 128), 0.05, np.float32)
    img += np.random.default_rng(0).normal(scale=0.005, size=img.shape).astype(np.float32)
    img[60:62, 40:64] = 0.9       # 2x24 bar: radius ~3.9 px, aspect ~13.8
    res = spots.detect(img)
    assert res and all(not s.accepted for s in res)
    assert {s.reason for s in res} == {"elongated"}
    img_v = np.full((128, 128), 0.05, np.float32)
    img_v += np.random.default_rng(0).normal(scale=0.005, size=img_v.shape).astype(np.float32)
    img_v[40:64, 60:62] = 0.9     # vertical canary: orientation invariance
    res_v = spots.detect(img_v)
    assert res_v and {s.reason for s in res_v} == {"elongated"}
