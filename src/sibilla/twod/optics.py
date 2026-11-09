"""Keyence channel optics constants for the 2D defocus pipeline.

Fixed imaging parameters of the widefield PCLS dataset:

- ``PIXEL_SIZE_UM``: camera pixel pitch projected to the sample plane (µm).
- ``NA``: objective numerical aperture.
- ``N_IM``: immersion refractive index of the sample space.

Assumption A1: ``N_IM = 1.33`` models an aqueous sample under an air
objective (the sample, not the objective, is the refractive medium the
emitted light propagates through). This affects only the absolute µm
labels of the pupil/PSF model, not the shape of the aberration fit.
"""

from dataclasses import dataclass

from sibilla.psf import PupilConfig

PIXEL_SIZE_UM: float = 1.887
NA: float = 0.20
N_IM: float = 1.33  # assumption A1: aqueous sample under air objective


@dataclass(frozen=True)
class ChannelOptics:
    """Per-channel optics of the Keyence imaging setup."""

    name: str
    lambda_nm: float
    kind: str


CHANNELS: dict[str, ChannelOptics] = {
    "CH1": ChannelOptics(name="CH1", lambda_nm=595.0, kind="fluorescence"),
    "CH4": ChannelOptics(name="CH4", lambda_nm=550.0, kind="transmission"),
}


def pupil_config(channel: str, grid_size: int = 256) -> PupilConfig:
    """Return the scalar pupil configuration for a Keyence channel.

    Raises KeyError for channels not in :data:`CHANNELS`.
    """
    return PupilConfig(
        pixel_size=PIXEL_SIZE_UM,
        lambda_um=CHANNELS[channel].lambda_nm / 1000.0,
        na=NA,
        n_im=N_IM,
        grid_size=grid_size,
    )
