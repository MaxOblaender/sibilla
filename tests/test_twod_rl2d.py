import numpy as np
import pytest
from scipy import fft as sp_fft
from sibilla.twod import rl2d
from sibilla.twod.psf2d import psf2d

def _blur(img, psf):
    return np.real(sp_fft.ifft2(sp_fft.fft2(img) * sp_fft.fft2(psf)))

def test_rl_reduces_spectral_error_with_correct_psf():
    rng = np.random.default_rng(0)
    obj = np.clip(rng.random((256, 256)), 0.02, 0.8).astype(np.float32)
    p = psf2d(12.0, "CH1", grid_size=256)
    blurred = _blur(obj, p) + 0.02
    res = rl2d.rl2d(blurred.astype(np.float32), p, background=0.02, n_iter=30)
    f = lambda x: np.abs(sp_fft.fft2(x - x.mean()))[40:80, 40:80].sum()
    assert f(res.image) > 1.3 * f(blurred)

def test_rl_with_wrong_psf_does_not_help():
    rng = np.random.default_rng(0)
    obj = np.clip(rng.random((256, 256)), 0.02, 0.8).astype(np.float32)
    right = psf2d(12.0, "CH1", grid_size=256)
    wrong = psf2d(45.0, "CH1", grid_size=256)
    blurred = _blur(obj, right) + 0.02
    good = rl2d.rl2d(blurred.astype(np.float32), right, background=0.02, n_iter=30)
    bad = rl2d.rl2d(blurred.astype(np.float32), wrong, background=0.02, n_iter=30)
    # spec §10: the correct PSF reduces the SPECTRUM ERROR vs the true
    # object; a wrong PSF does not. (TV was the wrong metric — with a
    # white-noise object, CORRECT RL legitimately has the HIGHER TV because
    # it restores the object's true high frequencies; measured gate round 1:
    # tv(bad)=2.9e4 vs tv(good)=2.4e5.)
    def spec_err(x):
        F = np.abs(sp_fft.fft2(x - x.mean()))
        return float(np.abs(F - np.abs(sp_fft.fft2(obj - obj.mean())))[10:40, 10:40].sum())
    assert spec_err(good.image) < spec_err(bad.image)

def test_empty_patch_stays_empty_with_background_subtracted():
    p = psf2d(20.0, "CH1", grid_size=128)
    flat = np.full((128, 128), 0.05, np.float32)
    res = rl2d.rl2d(flat, p, background=0.05, n_iter=20)
    assert res.image.std() < 0.01

def test_never_negative_and_converged_flag():
    rng = np.random.default_rng(2)
    img = np.clip(rng.random((128, 128)), 0, 1).astype(np.float32) + 0.05
    p = psf2d(6.0, "CH1", grid_size=128)
    res = rl2d.rl2d(img, p, background=0.05, n_iter=30)
    assert res.image.min() >= 0.0
    assert res.image.dtype == np.float32
    assert res.iterations <= 30

def test_early_stop_when_metric_plateaus():
    rng = np.random.default_rng(3)
    img = np.clip(rng.random((128, 128)), 0, 1).astype(np.float32) + 0.05
    p = psf2d(6.0, "CH1", grid_size=128)
    res = rl2d.rl2d(img, p, background=0.05, n_iter=30, metric=lambda x: 1.0)
    assert res.converged
    assert res.iterations < 30

def test_pedestal_is_not_amplified_into_texture():
    p = psf2d(15.0, "CH1", grid_size=128)
    empty = np.zeros((128, 128), np.float32)
    with_ped = empty + 0.3
    # pedestal shot noise: a noiseless pedestal cannot discriminate a
    # subtract-then-deconvolve implementation (measured: correct path max
    # 0.047 < 0.05, subtract-first 0.052 — fails the bound)
    with_ped = with_ped + np.random.default_rng(5).normal(
        scale=0.01, size=with_ped.shape).astype(np.float32)
    res = rl2d.rl2d(with_ped, p, background=0.3, n_iter=25)
    assert res.image.max() < 0.05
