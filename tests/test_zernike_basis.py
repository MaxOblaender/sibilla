"""Unit tests for the Noll Zernike basis (plan §5 tests: basis orthogonality).

Convention (pinned in plan §1 / README): Noll indexing, ORTHONORMAL over
the unit pupil (<Z_j, Z_k> = delta_jk), so wavefront RMS = sqrt(Σ a_j²)
for RMS-wave coefficients. Closed forms:
    Z_1 = 1
    m = 0 : Z = sqrt(n+1) R_n^0(rho)
    m > 0 : Z = sqrt(2(n+1)) R_n^m(rho) (cos | sin)(m theta)
hence max|Z_j| <= sqrt(2(n+1)) (== sqrt(2) for the low-order m>0 modes
that peak at the edge, e.g. Z2..Z4; Z5/Z6 peak inside the pupil).
"""

import math

import numpy as np
import pytest

from sibilla.zernike import (
    noll_basis,
    noll_nm,
    pupil_grid,
)


@pytest.mark.parametrize("j,expected", [
    (1, (0, 0, "c")), (2, (1, 1, "s")), (3, (1, 1, "c")), (4, (2, 0, "c")),
    (5, (2, 2, "c")), (6, (2, 2, "s")), (7, (3, 1, "s")), (8, (3, 1, "c")),
    (9, (3, 3, "s")), (10, (3, 3, "c")), (11, (4, 0, "c")), (12, (4, 2, "c")),
    (13, (4, 2, "s")), (14, (4, 4, "c")), (15, (4, 4, "s")),
])
def test_noll_mapping(j, expected):
    assert noll_nm(j) == expected


def test_basis_orthonormal_over_pupil():
    """<Z_j, Z_k> = delta_jk over the unit pupil (discrete, ~1e-3 tol)."""
    grid = 256
    rho, theta, mask = pupil_grid(grid)
    area = mask.sum()
    js = list(range(1, 13))
    Z = noll_basis(js, rho, theta)          # (J, M, N)
    gram = (Z[:, mask] @ Z[:, mask].T) / area
    # discrete-grid tolerance: area ~ pi/4 * G^2/4, boundary/aliasing error ~1e-3
    assert np.allclose(gram, np.eye(len(js)), atol=0.01), \
        f"max deviation={np.abs(gram - np.eye(len(js))).max()}"


def test_rms_wavefront_identity():
    """With orthonormal modes, the numerical wavefront RMS equals
    sqrt(sum a_j^2) for random coefficients (guards psf/metrics usage)."""
    rho, theta, mask = pupil_grid(256)
    area = mask.sum()
    js = list(range(1, 9))
    Z = noll_basis(js, rho, theta)
    rng = np.random.default_rng(0)
    a = rng.normal(size=len(js))
    w = (Z * a[:, None, None]).sum(axis=0)
    num = np.sqrt((w ** 2)[mask].sum() / area)
    ana = float(np.sqrt((a ** 2).sum()))
    assert num == pytest.approx(ana, rel=0.005), f"num={num} ana={ana}"


def test_known_shapes():
    """Canonical closed forms for Z1..Z8 (orthonormal scaling)."""
    rho, theta, mask = pupil_grid(64)
    def z(j):
        return noll_basis(j, rho, theta)[mask]
    assert np.allclose(z(1), 1.0)
    # Z2 = 2 rho sin(theta)   (tilt y; n=1, m=1 -> C = sqrt(2(n+1)) = 2)
    assert np.allclose(z(2), 2.0 * rho[mask] * np.sin(theta[mask]), atol=1e-12)
    # Z3 = 2 rho cos(theta)   (tilt x)
    assert np.allclose(z(3), 2.0 * rho[mask] * np.cos(theta[mask]), atol=1e-12)
    # Z4 = sqrt(3)(2rho^2 - 1)  (defocus; n=2, m=0 -> C = sqrt(n+1) = sqrt(3))
    assert np.allclose(z(4), math.sqrt(3.0) * (2 * rho[mask] ** 2 - 1), atol=1e-12)
    # Z5 = sqrt(6) rho^2 cos(2theta) (astig 0/90)
    assert np.allclose(z(5), math.sqrt(6.0) * rho[mask] ** 2
                       * np.cos(2 * theta[mask]), atol=1e-12)
    # Z6 = sqrt(6) rho^2 sin(2theta) (astig 45)
    assert np.allclose(z(6), math.sqrt(6.0) * rho[mask] ** 2
                       * np.sin(2 * theta[mask]), atol=1e-12)
    # Z7 = sqrt(8) (3rho^3 - 2rho) sin(theta)  (coma y; R_3^1 = 3rho^3 - 2rho)
    r3m1 = 3.0 * rho[mask] ** 3 - 2.0 * rho[mask]
    assert np.allclose(z(7), math.sqrt(8.0) * r3m1 * np.sin(theta[mask]), atol=1e-12)
    # Z8 = sqrt(8) (3rho^3 - 2rho) cos(theta)  (coma x)
    assert np.allclose(z(8), math.sqrt(8.0) * r3m1 * np.cos(theta[mask]), atol=1e-12)


def test_outside_pupil_is_zero():
    rho, theta, mask = pupil_grid(64)
    Z = noll_basis(4, rho, theta)
    assert np.allclose(Z[~mask], 0.0)


def test_single_vs_sequence_consistent():
    rho, theta, _ = pupil_grid(64)
    for j in (5, 6, 7, 8):
        a = noll_basis(j, rho, theta)
        b = noll_basis([j, j + 1], rho, theta)
        assert np.allclose(a, b[0])


def test_unsupported_index_raises():
    with pytest.raises(ValueError):
        noll_nm(1000)


def test_max_amplitude_bounds():
    """max|Z_j| <= sqrt(2(n+1)) for m>0, sqrt(n+1) for m=0 (edge/interior
    maxima never exceed the closed-form constant)."""
    rho, theta, mask = pupil_grid(256)
    for j in range(2, 17):
        n, m, _ = noll_nm(j)
        bound = math.sqrt(n + 1) if m == 0 else math.sqrt(2 * (n + 1))
        Z = noll_basis(j, rho, theta)
        assert np.abs(Z[mask]).max() <= bound + 1e-9
