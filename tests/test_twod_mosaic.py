import numpy as np
import pytest
from sibilla.twod import mosaic

def test_assemble_places_tiles_at_offsets():
    a = np.ones((8, 8), np.float32) * 0.2
    b = np.ones((8, 8), np.float32) * 0.6
    out = mosaic.assemble([a, b], [(0, 0), (0, 6)], (8, 14), blend_px=2)
    assert out.shape == (8, 14)
    assert out[4, 1] == pytest.approx(0.2, abs=0.01)
    assert out[4, 12] == pytest.approx(0.6, abs=0.01)

def test_assemble_blends_overlap_linearly():
    a = np.ones((8, 8), np.float32) * 0.2
    b = np.ones((8, 8), np.float32) * 0.6
    out = mosaic.assemble([a, b], [(0, 0), (0, 6)], (8, 14), blend_px=2)
    assert 0.2 < out[4, 6] < 0.6

def test_side_by_side_writes_png(tmp_path):
    before = np.linspace(0, 1, 64 * 64, dtype=np.float32).reshape(64, 64)
    after = before * 1.5
    p = tmp_path / "cmp.png"
    mosaic.side_by_side(before, after, p)
    assert p.exists() and p.stat().st_size > 0

def test_side_by_side_uses_one_shared_stretch(tmp_path):
    a = np.zeros((32, 32), np.float32)
    b = np.ones((32, 32), np.float32) * 0.5
    p = tmp_path / "cmp_shared.png"
    mosaic.side_by_side(a, b, p)
    from PIL import Image
    arr = np.asarray(Image.open(p).convert("L"))
    left = arr[:, : arr.shape[1] // 2]
    right = arr[:, arr.shape[1] // 2 :]
    # both panels clipped by the same window: they must not each be
    # normalised to their own [0, 255] range
    assert not (left.max() == 255 and right.max() == 255)

def test_three_panel_comparison(tmp_path):
    a = np.zeros((32, 32), np.float32)
    b = a + 0.3
    c = a + 0.6
    p = tmp_path / "triple.png"
    mosaic.panels_side_by_side([a, b, c], p)
    from PIL import Image
    arr = np.asarray(Image.open(p).convert("L"))
    assert arr.shape[1] >= 3 * 32   # three panels side by side
    w = arr.shape[1] // 3
    # one shared stretch: panel intensities stay ordered left-to-right
    assert arr[:, :w].mean() < arr[:, w : 2 * w].mean() < arr[:, 2 * w :].mean()
