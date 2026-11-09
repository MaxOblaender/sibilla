import numpy as np
import pytest
from sibilla.twod.psf2d import airy_first_zero_um, psf2d

CH = "CH1"

def test_energy_normalized_and_nonnegative():
    p = psf2d(0.0, CH, grid_size=128)
    assert p.shape == (128, 128)
    assert p.sum() == pytest.approx(1.0, rel=1e-10)
    assert p.min() >= 0.0

def test_airy_first_zero_diameter():
    # first dark ring DIAMETER = 1.22 * lambda / NA (µm); 2.44λ/NA would be 2×
    assert airy_first_zero_um(CH) == pytest.approx(1.22 * 0.595 / 0.20, rel=1e-6)

def test_symmetric_under_sign_of_dz():
    a = psf2d(+5.0, CH, grid_size=128)
    b = psf2d(-5.0, CH, grid_size=128)
    np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-12)

def test_encircled_energy_radius_grows_monotonically_with_defocus():
    # half-max FWHM is sub-pixel at Keyence sampling (r_half ≈ 0.5 px) — an
    # integer-sample half-max metric saturates. Encircled energy is robust.
    def _r_encircled(p, frac):
        G = p.shape[0]
        yy, xx = np.mgrid[0:G, 0:G]
        r = np.hypot(yy - G // 2, xx - G // 2)
        order = np.argsort(r.ravel())
        r_sorted = r.ravel()[order]
        e_sorted = p.ravel()[order]
        c = np.cumsum(e_sorted) / e_sorted.sum()
        return float(r_sorted[np.searchsorted(c, frac)])

    radii = [_r_encircled(psf2d(dz, CH, grid_size=128), 0.9)
             for dz in (0.0, 10.0, 20.0, 40.0, 60.0)]
    assert all(b > a for a, b in zip(radii, radii[1:]))
    # r80 saturates at the 1-px shell for dz <= 10 (measured, gate round 2);
    # r90 resolves the low-dz differences. Bounds are scale-free where possible:
    assert radii[0] < 2.0 and radii[-1] > 2 * radii[0]

def test_peak_falls_with_defocus_strehl_like():
    peaks = [psf2d(dz, CH, grid_size=128).max() for dz in (0.0, 10.0, 40.0)]
    assert peaks[0] > peaks[1] > peaks[2]

def test_coeffs_are_accepted_and_change_the_psf():
    clean = psf2d(0.0, CH, grid_size=128)
    aberrated = psf2d(0.0, CH, grid_size=128, coeffs={5: 0.15, 7: 0.10})
    assert not np.allclose(clean, aberrated)
    assert aberrated.max() < clean.max()
