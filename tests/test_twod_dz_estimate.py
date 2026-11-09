import numpy as np
import pytest
from scipy import fft as sp_fft
from sibilla.twod import dz_estimate as dzm, flatfield
from sibilla.twod.psf2d import psf2d

CH = "CH1"

def _object(seed=0, n=512):
    rng = np.random.default_rng(seed)
    # smooth-ish texture: low-passed noise + a few blobs (tissue-like)
    noise = rng.normal(size=(n, n))
    f = sp_fft.fft2(noise)
    fy = sp_fft.fftfreq(n)[:, None]
    fx = sp_fft.fftfreq(n)[None, :]
    f = f * np.exp(-((fx**2 + fy**2) / (2 * 0.12**2)))
    tex = np.real(sp_fft.ifft2(f))
    tex = (tex - tex.min()) / (tex.max() - tex.min())
    return (0.05 + 0.6 * tex).astype(np.float32)

def _blur(obj, dz_um):
    p = psf2d(dz_um, CH, grid_size=512)
    return np.real(sp_fft.ifft2(sp_fft.fft2(obj) * sp_fft.fft2(p))).astype(np.float32)

@pytest.mark.parametrize("true_dz", [0.0, 10.0, 20.0, 35.0])
def test_recovers_known_defocus(true_dz):
    obj = _object()
    ref = obj
    target = _blur(obj, true_dz)
    rng = np.random.default_rng(1)
    target = target + rng.normal(scale=0.002, size=target.shape).astype(np.float32)
    bins = dzm.radial_bins(target.shape)
    freqs, _ = dzm.radial_spectrum(target, bins)
    s_obj = dzm.radial_spectrum(ref, bins)[1]
    res = dzm.estimate_dz(target, ref, s_obj, freqs, CH, noise_sigma=0.002)
    assert not res.unrecoverable
    assert res.dz_um == pytest.approx(true_dz, abs=max(3.0, 0.15 * true_dz + 1.0))

def test_monotonic_in_true_defocus():
    obj = _object(3)
    bins = dzm.radial_bins(obj.shape)
    freqs, _ = dzm.radial_spectrum(obj, bins)
    s_obj = dzm.radial_spectrum(obj, bins)[1]
    est = [dzm.estimate_dz(_blur(obj, d), obj, s_obj, freqs, CH, noise_sigma=1e-4).dz_um
           for d in (0.0, 10.0, 20.0, 30.0)]
    assert all(b > a for a, b in zip(est, est[1:]))

def test_flat_background_is_unrecoverable():
    flat = np.full((512, 512), 0.05, np.float32)
    bins = dzm.radial_bins(flat.shape)
    freqs, _ = dzm.radial_spectrum(flat, bins)
    s_obj = np.ones_like(freqs)
    res = dzm.estimate_dz(flat, flat, s_obj, freqs, CH, noise_sigma=0.002)
    assert res.unrecoverable

def test_min_dz_identification_flags_dz0_as_indeterminate():
    """Round-4 pin: with ``min_dz_identification`` large the forced dz = 0
    fit becomes "no opinion" (dz_um=nan, indeterminate=True), because the
    MTF² model is stationary in dz at the 0 bound — measured on this
    construction the fit returns dz = 1e-10 with dz_err = inf (singular
    Jacobian), so dz = 0 can never be excluded from it. The default
    0.0 keeps the historical behaviour: a number, not nan."""
    obj = _object()
    target = _blur(obj, 0.0)
    rng = np.random.default_rng(1)
    target = target + rng.normal(scale=0.002, size=target.shape).astype(np.float32)
    bins = dzm.radial_bins(target.shape)
    freqs, _ = dzm.radial_spectrum(target, bins)
    s_obj = dzm.radial_spectrum(obj, bins)[1]
    res = dzm.estimate_dz(target, obj, s_obj, freqs, CH, noise_sigma=0.002)
    assert not res.indeterminate
    assert np.isfinite(res.dz_um)
    res50 = dzm.estimate_dz(target, obj, s_obj, freqs, CH, noise_sigma=0.002,
                            min_dz_identification=50.0)
    assert res50.indeterminate
    assert np.isnan(res50.dz_um)
    assert not res50.unrecoverable  # own class, distinct from unrecoverable

def test_sharpness_ranks_in_focus_above_blurred():
    obj = _object(4)
    assert dzm.sharpness(obj) > dzm.sharpness(_blur(obj, 25.0))

def test_select_references_picks_sharpest_tile():
    obj = _object(5)
    tiles = [_blur(obj, 30.0), _blur(obj, 5.0), _blur(obj, 25.0)]
    ref = dzm.select_references(tiles, n_tiles=1, patch=256)
    assert ref.shape == (256, 256)
    assert dzm.sharpness(ref) > dzm.sharpness(tiles[0])
