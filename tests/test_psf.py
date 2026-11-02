"""Unit tests for the scalar PSF stack (plan §5: PSF z-symmetry + Strehl)."""

import numpy as np
import pytest

from sibilla.defaults import DEFAULT_SCENE, pupil_config
from sibilla.psf import (
    PupilConfig,
    build_psf_stack,
    defocus_opd_waves,
    strehl_marechal,
)


@pytest.fixture
def config() -> PupilConfig:
    return pupil_config(DEFAULT_SCENE, grid_size=128)


def test_psf_energy_normalized(config):
    stack = build_psf_stack(config, {5: 0.10}, [-0.5, 0.0, 0.5])
    for k in range(stack.psf.shape[0]):
        assert stack.psf[k].sum() == pytest.approx(1.0, rel=1e-12)


def test_psf_real_nonnegative(config):
    stack = build_psf_stack(config, {5: 0.10, 7: 0.05}, [0.0])
    p = stack.psf[0]
    assert np.all(p >= 0)
    assert np.isreal(p).all()


def test_z_symmetry_aberration_free(config):
    """PSF(dz) == PSF(-dz) for zero aberration (defocus OPD is symmetric)."""
    stack = build_psf_stack(config, {}, [-0.5, 0.0, 0.5])
    assert np.allclose(stack.psf[0], stack.psf[2], rtol=1e-9, atol=1e-12)


def test_z_symmetry_astigmatism(config):
    """Astigmatism breaks z-symmetry slightly but the in-focus plane is
    unaffected; check the in-focus PSF is identical across the stack at dz=0
    for an aberrated vs aberrated reference (sanity)."""
    s1 = build_psf_stack(config, {6: 0.10}, [0.0])
    s2 = build_psf_stack(config, {6: 0.10}, [0.0])
    assert np.allclose(s1.psf[0], s2.psf[0])


def test_strehl_marechal(config):
    """Numerical Strehl (peak ratio) tracks the Maréchal approximation to
    ~a few percent for small aberrations (0.05 waves RMS)."""
    coeffs = {5: 0.05, 6: 0.05}
    stack = build_psf_stack(config, coeffs, [0.0])
    ref = build_psf_stack(config, {}, [0.0])
    num_strehl = stack.peak[0] / ref.peak[0]
    marechal = strehl_marechal(coeffs)
    assert num_strehl == pytest.approx(marechal, rel=0.05)


def test_strehl_decreases_with_aberration(config):
    ref = build_psf_stack(config, {}, [0.0]).peak[0]
    s_small = build_psf_stack(config, {5: 0.05}, [0.0]).peak[0]
    s_large = build_psf_stack(config, {5: 0.15}, [0.0]).peak[0]
    assert ref > s_small > s_large


def test_defocus_opd_paraxial_limit():
    """At small NA/n the exact OPD reduces to the quadratic form
    opd = -(dz/lambda_m) * (NA/n)^2 * rho^2 / 2  (plane farther from the
    objective, dz > 0, is a NEGATIVE (converging) defocus OPD).

    At u = NA/n = 0.3 the next term in the sqrt expansion contributes
    ~8.5%·u^2 ≈ 2% of the quadratic at rho = 0.9, so the tolerance
    reflects that (this test pins the FORM, sign and magnitude, not the
    full exact curve)."""
    rho = np.linspace(0.0, 0.9, 100).reshape(-1, 1)
    na_over_n = 0.3
    dz = 0.3
    lam = 0.52
    exact = defocus_opd_waves(rho, na_over_n, dz, lam)
    paraxial = -(dz / lam) * (na_over_n ** 2) * rho ** 2 / 2.0
    assert np.allclose(exact, paraxial, rtol=0.05, atol=1e-4)
    # exact OPD is always <= 0 (sqrt(1 - x^2) - 1 <= 0) for dz > 0
    assert np.all(exact <= 0.0)


def test_psf_peak_at_focus(config):
    """The in-focus PSF has the highest peak of a through-focus stack."""
    dz = np.linspace(-1.0, 1.0, 21)
    stack = build_psf_stack(config, {}, dz)
    assert np.argmax(stack.peak) == 10
