"""2D PSF per defocus plane, reusing the scalar widefield PSF path.

The 2D defocus pipeline does not model the PSF itself; it consumes the
pinned scalar widefield model (:func:`sibilla.psf.build_psf_stack`) one
plane at a time, so every convention (exact geometric defocus OPD,
energy normalization, shared pixel grid) is inherited unchanged and
validated here against analytic Airy geometry.
"""

from __future__ import annotations

import numpy as np

from sibilla.psf import build_psf_stack

from .optics import pupil_config


def psf2d(dz_um: float, channel: str, grid_size: int = 256,
          coeffs: dict[int, float] | None = None) -> np.ndarray:
    """Energy-normalized 2D PSF at defocus ``dz_um`` (µm) for a channel.

    Computed with the shared scalar widefield model in object space
    (``PupilConfig.pixel_size`` applies to both pupil grid and image, so
    lengths are sample-plane µm). The DC peak returned by
    :func:`sibilla.psf.build_psf_stack` sits at ``[0, 0]``; the array is
    rolled so the peak sits at ``[grid_size // 2, grid_size // 2]``.
    The roll is a permutation and does not change the sum, which is
    re-normalized defensively to exactly 1.

    coeffs : {Noll index: amplitude in RMS waves} (any subset), as in
    :func:`sibilla.psf.build_psf_stack`.
    """
    stack = build_psf_stack(pupil_config(channel, grid_size),
                            coeffs or {}, [dz_um])
    p = stack.psf[0]
    s = p.sum()
    if s > 0:
        p = p / s
    half = grid_size // 2
    return np.roll(p, (half, half), axis=(0, 1))


def airy_first_zero_um(channel: str) -> float:
    """Airy first dark-ring DIAMETER ``1.22 * lambda / NA`` in object space, µm.

    Convention corrected 2026-11-09 after the Task-2 gate: the first dark
    ring sits at radius ``1.22 * lambda / (2 * NA)``, i.e. diameter
    ``1.22 * lambda / NA``. The previously used ``2.44 * lambda / NA`` is
    the f-number formula misapplied to NA and is 2x too large.
    """
    cfg = pupil_config(channel)
    return 1.22 * cfg.lambda_um / cfg.na
