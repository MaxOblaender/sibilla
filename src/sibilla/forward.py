"""Forward model: object convolved with a per-plane PSF stack + background + Poisson.

Conventions (pinned in plans/zernike-tubule-test-plan.md §1/§5 and README.md):

- Scene approximation (planar object):
      image_j = O ⊛ PSF_j + B          [photons]
  where O is the (single-plane) object in photons, PSF_j the energy-
  normalized PSF at plane j, and B a constant per-pixel background offset
  in photons (shared across the stack, one global value in v1).
- Data are in photons. The *object* O is the *total photon budget* of each
  tubule (so that O ⊛ PSF_j conserves the per-tubule integrated photons on
  plane j). A per-tubule scalar amplitude scales that budget.
- Poisson shot noise is added on top of (O ⊛ PSF_j + B).
- The fitter sees uint16 data; the photon scale is documented in meta.json
  so that the fitter can work in photon units (we fit in photon space).

Key discrete-convolution convention (used by the fitter and tests): because
the pupil and the image share the same pixel size and the pupil is inscribed
in an FFT grid of the SAME shape as the image, the blur is the *cyclic*
convolution O ⋆ P on that shared grid:  ifft2(fft2(O) · fft2(P)). This equals
true linear convolution wherever the object and its blur stay inside the
frame — guaranteed for tubules kept >= border_margin_um (1.0 µm) from the
edge, since the PSF support (~5 px core + negligible skirt) is smaller than
the margin; any wrapped skirt energy is many orders of magnitude below the
detector noise and lands far from every tubule. Cyclic convolution also
conserves total photons exactly: sum(O ⋆ P) = sum(O)·sum(P) = sum(O) for an
energy-normalized PSF.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy import fft as sp_fft

from .psf import PupilConfig, PupilStack, build_confocal_psf_stack, build_psf_stack


@dataclass
class ImageData:
    """A noisy image stack and its provenance."""
    data: np.ndarray                # (Nz, M, N) float, photons (before uint16 cast)
    data_uint16: np.ndarray         # (Nz, M, N) uint16
    background: float               # photons/px (true value)
    z_planes_um: np.ndarray         # (Nz,) plane defocus, µm
    photon_scale: float = 1.0       # 1.0 => data already in photons


def cyclic3d(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """3D cyclic (FFT) convolution of two same-shape volumes.

    (a ⊛ b)[i,j,k] = sum a[i',j',k'] b[(i-i')..]; in the frequency domain
    ifftn(fftn(a) · fftn(b)). Used for the 3D forward model where the PSF is a
    through-focus volume (index 0 = in-focus, cyclic in z) and the object is a
    3D footprint.
    """
    return sp_fft.ifftn(sp_fft.fftn(a) * sp_fft.fftn(b)).real


def psf3d_cyclic(config, coeffs: dict, nz: int, z_step_um: float,
                 model: str = "widefield",
                 lambda_exc_um: float = 0.493) -> np.ndarray:
    """Through-focus PSF volume (nz, M, N) in cyclic-z convention.

    Index 0 = in-focus; index i = defocus i·z_step (i <= half) or
    (i - nz)·z_step (i > half). This is exactly the kernel for `cyclic3d`
    with a 3D object, so image[k] = sum_k' O[k'] · PSF(k - k').

    model: "widefield" (single detection path) or "confocal" (spinning-disk
    product PSF: excitation × detection, same Zernikes at both wavelengths).
    """
    half = (nz - 1) / 2.0
    defo = [(i if i <= half else i - nz) * z_step_um for i in range(nz)]
    if model == "confocal":
        return build_confocal_psf_stack(config, coeffs, defo,
                                        lambda_exc_um=lambda_exc_um).psf
    return build_psf_stack(config, coeffs, defo).psf


def convolve_plane(object_plane: np.ndarray, psf_plane: np.ndarray) -> np.ndarray:
    """Cyclic convolution of an object plane with a PSF plane on a shared grid.

    Both inputs must have the SAME shape (the plan's shared-grid convention:
    the pupil/FFT grid is the same size as the image). The PSF from
    `psf.build_psf_stack` is |ifft2(pupil)|² of an unshifted pupil, i.e.
    zero-centered (peak at index [0, 0], wings wrapping to the far corner) —
    which is exactly the right convention for a cyclic convolution, so NO
    shift is needed. See the module docstring for why cyclic == linear here.
    """
    if object_plane.shape != psf_plane.shape:
        raise ValueError(
            f"shared-grid convention violated: object {object_plane.shape} "
            f"vs PSF {psf_plane.shape} (build the PSF stack at the image grid size)")
    return sp_fft.ifft2(sp_fft.fft2(object_plane) * sp_fft.fft2(psf_plane)).real


class FastBlur:
    """Precomputed-FFT blurring, exactly equivalent to `convolve_plane`.

    Precomputes the FFT of the object once; each PSF plane then costs one
    FFT + one pointwise multiply + one IFFT — several times cheaper than a
    fresh FFT pair per (object, PSF) evaluation, which the fitter does
    ~200 times per start.
    """

    def __init__(self, object_plane: np.ndarray):
        self.shape = object_plane.shape
        self.fo = sp_fft.fft2(object_plane)

    def blur(self, psf_plane: np.ndarray) -> np.ndarray:
        if self.shape != psf_plane.shape:
            raise ValueError(
                f"shared-grid convention violated: object {self.shape} "
                f"vs PSF {psf_plane.shape}")
        return sp_fft.ifft2(self.fo * sp_fft.fft2(psf_plane)).real


def forward_stack(object_plane: np.ndarray, psf: PupilStack,
                  amplitude: float = 1.0, background: float = 0.0,
                  noise: bool = False, rng: np.random.Generator | None = None,
                  z_offset_um: float = 0.0,
                  z_planes_um: Sequence[float] | None = None) -> np.ndarray:
    """(O*amp) ⊛ PSF_j + B for each plane j, with optional Poisson noise.

    Parameters
    ----------
    object_plane : (M, N) photons, the single in-focus object (sum per tubule).
    psf : energy-normalized PSF stack over z_planes_um.
    amplitude : scalar multiplier on the object photon budget (per tubule in
        the fitter; 1.0 for the generator).
    background : photons/px constant offset.
    noise : if True and rng given, add Poisson shot noise to the result.
    z_offset_um : shift the *observed* plane defocus by this amount (used by
        the fitter's shared z-offset nuisance). The PSF is evaluated at
        (z_planes_um + z_offset_um).
    z_planes_um : explicit defocus grid to use (defaults to psf.z_planes_um).
        When z_offset_um != 0 the PSF must be re-evaluated at the shifted
        planes; callers that need that should pre-build a shifted stack.
        Here we simply select/evaluate the provided `psf` as-is at its own
        planes, and treat z_offset_um as a *label* — the actual shifted
        stack is built by the caller (see fit.py). Kept for API clarity.

    Returns (Nz, M, N) float in photons.
    """
    planes = psf.z_planes_um
    out = np.empty(planes.shape + object_plane.shape)
    o = object_plane * amplitude
    for k in range(len(planes)):
        im = convolve_plane(o, psf.psf[k]) + background
        out[k] = im
    if noise and rng is not None:
        out = rng.poisson(out)
    return out


def render_uint16(photon_stack: np.ndarray, photon_scale: float = 1.0) -> np.ndarray:
    """Cast a photon stack to uint16, clipping to the dtype range.

    The photon_scale documents how to recover photons from the stored
    uint16 values: photons = uint16_value * photon_scale. In v1 we keep
    photon_scale = 1.0 and clip to 65535 (the scenes here peak well below
    that; see make_dataset).
    """
    a = np.clip(np.round(photon_stack), 0, 65535)
    return a.astype(np.uint16)


def energy_conserved(object_plane: np.ndarray, psf_plane: np.ndarray,
                     tol: float = 1e-6) -> bool:
    """True iff sum(O ⊛ PSF) ≈ sum(O) * sum(PSF) (within tol).

    For an energy-normalized PSF (sum=1) this checks that convolution
    conserves the object's integrated photons.
    """
    im = convolve_plane(object_plane, psf_plane)
    lhs = im.sum()
    rhs = object_plane.sum() * psf_plane.sum()
    return bool(np.isclose(lhs, rhs, rtol=tol, atol=tol * max(1.0, abs(rhs))))
