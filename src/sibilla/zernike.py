"""Noll orthonormal Zernike basis on a circular pupil grid.

Conventions (pinned in plans/zernike-tubule-test-plan.md §1 and README.md):

- Noll indexing, modes 1..N (piston = Z1, tilt_y = Z2, tilt_x = Z3,
  defocus = Z4, ...). For each order n the sequence within the block is
  the standard Noll one: m = 0 (even n) first, then each m paired as
  (cos, sin) for even n and (sin, cos) for odd n. See _NOLL_TABLE.
- RMS-normalized (orthonormal) over the unit pupil: <Z_j, Z_k> = delta_jk,
  so the wavefront RMS = sqrt(sum_j a_j^2) when coefficients are in RMS
  waves. This is achieved with Z_1 = 1 and, for all other modes,
  Z_j = sqrt(2) * R_n^m(rho) * {cos, sin}(m theta), which has max |Z_j| =
  sqrt(2) (the alternative max-normalization would make the basis
  non-orthonormal and break the RMS identity).
- Coefficients are in RMS waves.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

#: Noll mode names for the low-order set.
NOLL_NAMES: dict[int, str] = {
    1: "piston",
    2: "tilt_y",
    3: "tilt_x",
    4: "defocus",
    5: "astig_090",
    6: "astig_45",
    7: "coma_y",
    8: "coma_x",
    9: "trefoil_1",
    10: "trefoil_2",
    11: "sph3",
    12: "sph2",
    13: "sph2b",
    14: "quadra_x",
    15: "quadra_y",
    16: "sph4",
}


def _noll_sequence(n_max: int) -> list[tuple[int, int, str]]:
    """(n, m, 'c'/'s') in Noll order, Z1..Z_{(n_max+1)(n_max+2)/2}."""
    seq: list[tuple[int, int, str]] = [(0, 0, "c")]
    for n in range(1, n_max + 1):
        if n % 2 == 1:
            ms = list(range(1, n + 1, 2))
        else:
            ms = [0] + list(range(2, n + 1, 2))
        for m in ms:
            if m == 0:
                seq.append((n, 0, "c"))
            elif n % 2 == 1:
                seq.append((n, m, "s"))
                seq.append((n, m, "c"))
            else:
                seq.append((n, m, "c"))
                seq.append((n, m, "s"))
    return seq


# Derived table (Z1..Z35), guarded against the canonical Noll 1976 sequence.
_NOLL_SEQ = _noll_sequence(8)
_NOLL_TABLE: dict[int, tuple[int, int, str]] = {
    j + 1: t for j, t in enumerate(_NOLL_SEQ)
}
_CANONICAL: dict[int, tuple[int, int, str]] = {
    1: (0, 0, "c"), 2: (1, 1, "s"), 3: (1, 1, "c"), 4: (2, 0, "c"),
    5: (2, 2, "c"), 6: (2, 2, "s"), 7: (3, 1, "s"), 8: (3, 1, "c"),
    9: (3, 3, "s"), 10: (3, 3, "c"), 11: (4, 0, "c"), 12: (4, 2, "c"),
    13: (4, 2, "s"), 14: (4, 4, "c"), 15: (4, 4, "s"), 16: (5, 1, "s"),
    17: (5, 1, "c"), 18: (5, 3, "s"), 19: (5, 3, "c"), 20: (5, 5, "s"),
    21: (5, 5, "c"), 22: (6, 0, "c"), 23: (6, 2, "c"), 24: (6, 2, "s"),
    25: (6, 4, "c"), 26: (6, 4, "s"), 27: (6, 6, "c"), 28: (6, 6, "s"),
}
for _j, _t in _CANONICAL.items():
    assert _NOLL_TABLE[_j] == _t, f"Noll mapping wrong for Z{_j}: {_NOLL_TABLE[_j]} != {_t}"


def noll_nm(j: int) -> tuple[int, int, str]:
    """(n, m, kind) for Noll index j; kind 'c' -> cos(m theta), 's' -> sin."""
    if j not in _NOLL_TABLE:
        raise ValueError(f"unsupported Noll index {j} (table covers 1..{max(_NOLL_TABLE)})")
    return _NOLL_TABLE[j]


def mode_name(j: int) -> str:
    """Human-readable name for a Noll mode (falls back to 'Z{j}')."""
    return NOLL_NAMES.get(j, f"Z{j}")


def _radial_r(n: int, m: int, rho: np.ndarray) -> np.ndarray:
    """Radial polynomial R_n^m(rho), n >= |m|, n - |m| even."""
    if n < abs(m) or (n - abs(m)) % 2:
        raise ValueError(f"invalid (n, m) = ({n}, {m})")
    s = (n - abs(m)) // 2
    total = np.zeros_like(rho)
    for k in range(s + 1):
        num = (-1) ** k * math.factorial(n - k)
        den = (math.factorial(k)
               * math.factorial((n + abs(m)) // 2 - k)
               * math.factorial((n - abs(m)) // 2 - k))
        total = total + (num / den) * rho ** (n - 2 * k)
    return total


def noll_basis(indices: int | Sequence[int], rho: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Evaluate Noll Zernikes (orthonormal) at pupil coordinates.

    Parameters
    ----------
    indices : int or sequence of ints
        Noll index or indices.
    rho : (M, N) array
        Normalized radial coordinate (0 at center, 1 at the pupil edge).
        Values > 1 (outside the pupil) are masked to 0.
    theta : (M, N) array
        Azimuth in radians (0 along +x, counter-clockwise toward +y).

    Returns
    -------
    (M, N) for a single index, (J, M, N) for a sequence.
    """
    single = np.isscalar(indices)
    idxs = [indices] if single else list(indices)
    out = []
    mask = rho <= 1.0
    for j in idxs:
        n, m, kind = noll_nm(j)
        r = np.where(mask, rho, 0.0)
        R = _radial_r(n, m, r)
        if m == 0:
            # orthonormal: <Z,Z>_pupil = 1  ->  C = sqrt(n+1)
            val = math.sqrt(n + 1) * R
        else:
            # orthonormal: C = sqrt(2(n+1))
            trig = np.cos(m * theta) if kind == "c" else np.sin(m * theta)
            val = math.sqrt(2 * (n + 1)) * R * trig
        out.append(np.where(mask, val, 0.0))
    arr = np.stack(out, axis=0)
    return arr[0] if single else arr


def pupil_grid(shape: tuple[int, int] | int = 256) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Circular pupil grid on a square array.

    The pupil is inscribed in the square: radius = (max(shape) - 1) / 2 px.

    Returns
    -------
    rho : (M, N) normalized radial coordinate (0..1 inside the pupil)
    theta : (M, N) azimuth, radians (0 along +x, CCW toward +y)
    mask : (M, N) bool, True inside the pupil
    """
    if np.isscalar(shape):
        shape = (shape, shape)
    m, n = shape
    y, x = np.meshgrid(np.arange(m) - (m - 1) / 2.0,
                       np.arange(n) - (n - 1) / 2.0, indexing="ij")
    r = np.sqrt(x * x + y * y) / ((max(m, n) - 1) / 2.0)
    theta = np.arctan2(y, x)
    mask = r <= 1.0
    return r, theta, mask
