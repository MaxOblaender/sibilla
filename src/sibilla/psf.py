"""Scalar widefield PSF stack from Zernike coefficients.

Conventions (pinned in plans/zernike-tubule-test-plan.md §1 and README.md):

- Pupil phase at defocus dz (µm), for a plane displaced by dz from the
  object plane (dz > 0 = plane farther from the objective):

      phi = 2π [ Σ_j a_j Z_j  +  opd_defocus(dz) ]        (waves bracket)

  where a_j are in RMS waves (Noll, RMS-normalized) and the exact
  geometric (Gibson–Lanni style) defocus OPD in waves is

      opd_defocus(ρ; dz) = (dz / λ_m) ( sqrt(1 - (NA/n_im · ρ)²) - 1 )

  with λ_m = n_im · λ (wavelength in the medium) and ρ the normalized
  pupil radius. Paraxial limit: (dz/λ_m)·(NA/n_im)²·ρ²/2, the standard
  quadratic defocus term.

- PSF(dz) = |ifft2(pupil)|², energy-normalized (sums to 1), on a grid with
  the SAME pixel size as the image (PupilConfig.pixel_size). The pupil is
  inscribed in that grid (radius (G-1)/2 px), so the OTF frequency bins
  match the image FFT bins exactly and discrete circular convolution
  equals true linear convolution for objects smaller than the grid.

- The zero-aberration PSF is symmetric under dz -> -dz for the quadratic
  (paraxial) term; the exact term keeps the symmetry to well below
  double precision at NA = 1.4, n = 1.518 (see tests).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .zernike import noll_basis, pupil_grid


@dataclass(frozen=True)
class PupilConfig:
    """Scalar pupil / imaging configuration.

    All lengths in µm.
    """
    pixel_size: float          # µm per pixel (image AND pupil grid)
    lambda_um: float           # vacuum wavelength, µm
    na: float                  # numerical aperture
    n_im: float                # immersion refractive index
    grid_size: int = 256       # square pupil / FFT grid size (px)

    @property
    def lambda_m(self) -> float:
        """Wavelength in the immersion medium, µm."""
        return self.lambda_um * self.n_im

    @property
    def na_over_n(self) -> float:
        """NA / n_im = sin(theta) at the pupil edge (0 < x <= 1)."""
        return self.na / self.n_im

    @property
    def pupil_radius_px(self) -> float:
        return (self.grid_size - 1) / 2.0


def defocus_opd_waves(rho: np.ndarray, na_over_n: float, dz_um: float,
                      lambda_m_um: float) -> np.ndarray:
    """Exact geometric defocus OPD (waves) at defocus dz (µm).

    opd(ρ; dz) = (dz / λ_m) · (sqrt(1 - (na_over_n · ρ)²) - 1)

    Finite inside the pupil (|na_over_n·ρ| < 1 for NA < n_im).
    """
    s = na_over_n * rho
    arg = np.clip(1.0 - s * s, 0.0, None)
    return (dz_um / lambda_m_um) * (np.sqrt(arg) - 1.0)


@dataclass
class PupilStack:
    """Per-plane PSF stack and metadata."""
    psf: np.ndarray                      # (Nz, M, N), energy-normalized
    z_planes_um: np.ndarray              # (Nz,) defocus per plane, µm
    config: PupilConfig
    coeffs: dict[int, float] = field(default_factory=dict)

    @property
    def peak(self) -> np.ndarray:
        """PSF peak (max intensity) per plane."""
        return self.psf.max(axis=(1, 2))

    def strehl(self) -> np.ndarray:
        """Strehl per plane: peak / peak of the diffraction-limited PSF at
        the same defocus (aberration-free reference)."""
        ref = build_psf_stack(self.config, {}, self.z_planes_um).peak
        return self.peak / ref


def build_psf_stack(config: PupilConfig, coeffs: dict[int, float],
                    z_planes_um: Sequence[float]) -> PupilStack:
    """Energy-normalized scalar PSF stack for a set of Zernike coefficients.

    coeffs : {Noll index: amplitude in RMS waves} (any subset)
    z_planes_um : defocus per plane, µm (dz > 0 = farther from objective)
    """
    G = config.grid_size
    rho, theta, mask = pupil_grid(G)
    idxs = list(coeffs.keys())
    Z = None
    vals = None
    if coeffs:
        Z = noll_basis(idxs, rho, theta)                 # (J, M, N)
        vals = np.array([coeffs[j] for j in idxs], dtype=float)

    psf = np.empty((len(z_planes_um), G, G))
    for k, dz in enumerate(z_planes_um):
        w = np.zeros_like(rho)
        if Z is not None:
            w += (Z * vals[:, None, None]).sum(axis=0)
        w += defocus_opd_waves(rho, config.na_over_n, dz, config.lambda_m)
        w = np.where(mask, w, 0.0)
        pupil = np.where(mask, np.exp(2j * np.pi * w), 0.0)
        i = np.abs(np.fft.ifft2(pupil)) ** 2
        s = i.sum()
        if s > 0:
            i /= s
        psf[k] = i
    return PupilStack(psf=psf, z_planes_um=np.asarray(z_planes_um, float),
                      config=config, coeffs=dict(coeffs))


def strehl_marechal(coeffs: dict[int, float]) -> float:
    """Maréchal approximation of the in-focus Strehl: exp(-(2πσ)²) with
    σ = sqrt(Σ a_j²) (RMS waves). Cross-check for the numerical Strehl."""
    sigma = float(np.sqrt(sum(v * v for v in coeffs.values())))
    return float(np.exp(-(2.0 * np.pi * sigma) ** 2))


# --- Spinning-disk confocal product model (spec risk 1 v2, user 2026-10-20) ---

#: AF488 excitation wavelength, nm (CZI dye metadata; emission is in config)
DEFAULT_EXCITATION_NM: float = 493.0


def build_confocal_psf_stack(config: PupilConfig, coeffs: dict[int, float],
                             z_planes_um: Sequence[float],
                             lambda_exc_um: float = DEFAULT_EXCITATION_NM / 1000.0,
                             per_plane: bool = True
                             ) -> PupilStack:
    """Spinning-disk confocal PSF stack: excitation × detection product.

    The effective confocal PSF at ~1 AU pinhole is the product of the
    excitation PSF (λ_exc) and the detection PSF (λ_em = config.lambda_um),
    both formed by the SAME objective and therefore carrying the SAME
    Zernike wavefront (evaluated at their own wavelengths):

        PSF_confocal(Δz) ∝ |FFT(pupil(λ_exc))|² · |FFT(pupil(λ_em))|²

    Each factor is a `build_psf_stack` on the same grid (each per-plane
    normalized); with per_plane=True the product is re-normalized per plane
    (original behavior, used by the fitter). With per_plane=False the
    product's AXIAL ENVELOPE is kept intact: the per-plane energy then
    decays away from focus (the confocal sectioning signal). That variant
    is the physically correct 3D kernel for deconvolution — re-normalizing
    per plane erases the sectioning and leaves a rank-deficient z-system
    in which RL shuffles flux between adjacent planes (the bright-slice
    artifact, 2026-10-22). The caller normalizes globally afterwards.
    """
    exc = build_psf_stack(config, coeffs, z_planes_um, ) if lambda_exc_um == config.lambda_um else None
    if exc is None:
        cfg_exc = PupilConfig(
            pixel_size=config.pixel_size,
            lambda_um=lambda_exc_um,
            na=config.na,
            n_im=config.n_im,
            grid_size=config.grid_size,
        )
        exc = build_psf_stack(cfg_exc, coeffs, z_planes_um)
    det = build_psf_stack(config, coeffs, z_planes_um)
    psf = exc.psf * det.psf
    if per_plane:
        for k in range(psf.shape[0]):
            s = psf[k].sum()
            if s > 0:
                psf[k] /= s
    return PupilStack(psf=psf, z_planes_um=np.asarray(z_planes_um, float),
                      config=config, coeffs=dict(coeffs))
