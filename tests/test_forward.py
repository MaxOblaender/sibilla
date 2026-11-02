"""Unit tests for the forward model (cyclic convolution on a shared grid).

The PSF stack is built at the IMAGE grid size (plan §1: shared pixel size,
pupil inscribed in the image-sized FFT grid). The blur is the cyclic
convolution O ⋆ P = ifft2(fft2(O)·fft2(P)); for tubules kept >= 1 µm from
the frame edge this equals true linear convolution (see forward.py
docstring). Tests use a small 64-px frame for speed.
"""

import numpy as np
import pytest
from scipy.signal import fftconvolve

from sibilla.defaults import SceneSpec, pupil_config
from sibilla.forward import (
    FastBlur,
    convolve_plane,
    energy_conserved,
    forward_stack,
    render_uint16,
)
from sibilla.psf import build_psf_stack
from sibilla.scene import render_tubule_unit


def _spec(fov: int = 64) -> SceneSpec:
    return SceneSpec(fov_px=fov, pixel_size_um=0.1)


def _config(fov: int = 64):
    return pupil_config(_spec(fov), grid_size=fov)


def _linear_ref_zero_centered(o: np.ndarray, p: np.ndarray) -> np.ndarray:
    """True LINEAR blur of o by a zero-centered PSF p (peak at [0,0], wings
    wrapping into the far corner), computed independently via scipy's
    real-space fftconvolve.

    Builds an explicit (2g-1, 2g-1) kernel K whose index (w, w) (w = g-1)
    corresponds to offset 0: K[w+dy, w+dx] = p[dy mod g, dx mod g] — the
    positive-offset half from p directly and the negative-offset half from
    the wrapped corner. Then L[i, j] = (o * K)[w+i, w+j].
    """
    g = o.shape[0]
    w = g - 1
    oy, ox = np.mgrid[-w:g, -w:g]
    K = p[oy % g, ox % g]
    full = fftconvolve(o, K, mode="full")
    return full[w:w + g, w:w + g]


def test_convolve_plane_matches_full_convolution():
    """For compact, in-frame supports the cyclic convolution equals the
    top-left block of the true linear (full) convolution — calibrates the
    no-wrap-around regime used by the dataset."""
    rng = np.random.default_rng(0)
    m = n = 32
    # compact structured object, away from the bottom/right edges
    O = np.zeros((m, n))
    O[10:22, 8:24] = rng.random((12, 16))
    # zero-centered PSF (peak at [0,0], small support, asymmetric, no wrap)
    yy, xx = np.mgrid[0:6, 0:7]
    core = np.exp(-((xx - 2.0) ** 2 + (yy - 1.0) ** 2) / 8.0)
    P = np.zeros((m, n))
    P[:6, :7] = core / core.sum()
    mine = convolve_plane(O, P)
    ref = fftconvolve(O, P, mode="full")[:m, :n]
    assert np.allclose(mine, ref, rtol=1e-10, atol=1e-12)


def test_cyclic_equals_linear_in_frame():
    """With the tubule kept well inside the frame (plan's 1 µm border
    margin), the cyclic blur equals the true linear convolution to ~1e-5
    relative (differences only in the negligible outer-Airy tails and the
    wrapped corner skirt, far from every tubule)."""
    spec, cfg = _spec(64), _config(64)
    stack = build_psf_stack(cfg, {6: 0.10}, [0.0])
    o = render_tubule_unit(16, 16, 48, 48, spec)
    mine = convolve_plane(o, stack.psf[0])
    ref = _linear_ref_zero_centered(o, stack.psf[0])
    rel = np.abs(mine - ref).max() / np.abs(mine).max()
    assert rel < 1e-5, f"cyclic vs linear max rel diff = {rel}"


def test_fastblur_matches_convolve_plane():
    spec, cfg = _spec(64), _config(64)
    stack = build_psf_stack(cfg, {5: 0.05, 8: 0.03}, [0.0, 0.25])
    o = render_tubule_unit(16, 20, 48, 40, spec)
    assert o.shape == stack.psf[0].shape
    fb = FastBlur(o)
    for k in range(2):
        assert np.allclose(fb.blur(stack.psf[k]), convolve_plane(o, stack.psf[k]),
                           rtol=1e-12, atol=1e-15)


def test_shape_mismatch_raises():
    spec, cfg = _spec(64), _config(64)
    stack = build_psf_stack(cfg, {}, [0.0])
    o_small = np.zeros((32, 32))
    o_small[16, 16] = 1.0
    with pytest.raises(ValueError):
        convolve_plane(o_small, stack.psf[0])
    with pytest.raises(ValueError):
        FastBlur(o_small).blur(stack.psf[0])


def test_convolution_energy_conserved():
    """Cyclic convolution conserves total photons EXACTLY
    (sum(O ⋆ P) = sum(O)·sum(P)), wrapped skirt included."""
    spec, cfg = _spec(64), _config(64)
    stack = build_psf_stack(cfg, {5: 0.05}, [0.0])
    o = render_tubule_unit(16, 16, 48, 48, spec)   # centered
    assert energy_conserved(o, stack.psf[0], tol=1e-9)
    im = convolve_plane(o, stack.psf[0])
    assert im.sum() == pytest.approx(o.sum(), rel=1e-9)


def test_forward_adds_background_exactly():
    """With a zero object, the forward output is exactly the background."""
    cfg = _config(64)
    stack = build_psf_stack(cfg, {}, [0.0, 0.25])
    o = np.zeros((64, 64))
    out = forward_stack(o, stack, amplitude=1.0, background=7.0, noise=False)
    assert np.allclose(out, 7.0)


def test_forward_total_photons_with_background():
    """sum(out) == amp·sum(O) + bg·Npx exactly (cyclic conservation,
    PSF energy 1, Poisson off)."""
    spec, cfg = _spec(64), _config(64)
    stack = build_psf_stack(cfg, {6: 0.05}, [0.0])
    m, n = 64, 64
    o = render_tubule_unit(n // 4, m // 4, 3 * n // 4, 3 * m // 4, spec)
    bg = 3.0
    out = forward_stack(o, stack, amplitude=2.5, background=bg, noise=False)
    expected = o.sum() * 2.5 + bg * m * n
    assert out.sum() == pytest.approx(expected, rel=1e-12)


def test_forward_poisson_noise_statistics():
    """With noise=True the output mean matches the noiseless expectation and
    var ≈ mean (Poisson) in a mid-level region. A 300 nm tubule (vs 60 nm in
    the real scene) is used so the blurred object spans many pixels — the
    statistic being tested (Poisson counting) is independent of diameter."""
    spec, cfg = _spec(64), _config(64)
    stack = build_psf_stack(cfg, {}, [0.0])
    o = render_tubule_unit(16, 16, 48, 48, spec, diameter_um=0.6)
    rng = np.random.default_rng(42)
    noisy = forward_stack(o, stack, amplitude=3000.0, background=2.0, noise=True, rng=rng)
    clean = forward_stack(o, stack, amplitude=3000.0, background=2.0, noise=False)
    assert noisy.shape == clean.shape
    assert np.isclose(noisy.mean(), clean.mean(), rtol=0.05)
    region = clean > 5.0
    assert region.sum() > 50
    assert np.isclose(noisy[region].var(), noisy[region].mean(), rtol=0.25)


def test_uint16_roundtrip():
    stack = np.array([[[1.0, 2.5], [70000.0, 3.0]]], dtype=float)
    u = render_uint16(stack)
    assert u.dtype == np.uint16
    assert u[0, 0, 0] == 1
    assert u[0, 1, 0] == 65535      # clipped
    assert u[0, 1, 1] == 3
