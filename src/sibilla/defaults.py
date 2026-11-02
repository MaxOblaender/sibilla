"""Shared physics and experiment defaults (plan §3, §6 — single source of truth)."""

from __future__ import annotations

from dataclasses import dataclass

from .psf import PupilConfig

# --- Physics (plan §3) -------------------------------------------------------

#: vacuum wavelength, nm
LAMBDA_NM: float = 520.0
#: numerical aperture
NA: float = 1.4
#: immersion refractive index
N_IM: float = 1.518


@dataclass(frozen=True)
class SceneSpec:
    fov_px: int = 256
    pixel_size_um: float = 0.1        # 100 nm pixels
    n_planes: int = 21
    z_step_um: float = 0.25           # 250 nm step
    n_segments: int = 10
    length_range_um: tuple[float, float] = (4.0, 7.0)
    min_sep_um: float = 1.0
    border_margin_um: float = 1.0
    max_angular_gap_deg: float = 100.0
    max_attempts: int = 20000
    # fallback (plan §2): relax to 0.8 µm or drop to 8 segments
    fallback_min_sep_um: float = 0.8
    fallback_n_segments: int = 8
    fluorescent_diameter_um: float = 0.060   # 60 nm (plan §4 caveat 1)
    supersample: int = 8


DEFAULT_SCENE: SceneSpec = SceneSpec()


def pupil_config(scene: SceneSpec = DEFAULT_SCENE,
                 grid_size: int | None = None) -> PupilConfig:
    """PupilConfig matching the scene's pixel grid (plan §1: sampling matched)."""
    return PupilConfig(
        pixel_size=scene.pixel_size_um,
        lambda_um=LAMBDA_NM / 1000.0,
        na=NA,
        n_im=N_IM,
        grid_size=grid_size or scene.fov_px,
    )


# --- Photometry / noise (plan §3) --------------------------------------------

#: peak-SNR levels (peak of the *object* in photons per pixel)
PEAK_SNR_LEVELS: tuple[float, ...] = (20.0, 50.0, 100.0)
DEFAULT_PEAK: float = 50.0
#: background ≈ 5% of peak, photons/px
BG_FRACTION: float = 0.05


def nominal_budget(peak: float, scene: SceneSpec = DEFAULT_SCENE) -> float:
    """Per-tubule photon budget B_nom such that amplitudes ≈ 1 at the true scene.

    B_nom = peak [photons/px] / pixels-per-µm² = peak · (px/µm)².
    """
    px_per_um = 1.0 / scene.pixel_size_um
    return peak * px_per_um * px_per_um


def background(peak: float) -> float:
    """Shared background offset, photons/px."""
    return BG_FRACTION * peak


# --- Injection cases (plan §3) -------------------------------------------------

NULL_COEFFS: dict[int, float] = {}
SINGLE_MODE_AMP: float = 0.10
COMBINED_COEFFS: dict[int, float] = {5: 0.10, 6: 0.10, 7: 0.07, 8: 0.07}
DEFOCUS_COEFFS: dict[int, float] = {4: 0.10}   # experiment 4 (leakage only)
AMP_SWEEP_Z6: tuple[float, ...] = (0.05, 0.10, 0.20)   # optional sweep

# --- Fitter (plan §5) ------------------------------------------------------------

FITTED_MODES: tuple[int, ...] = (5, 6, 7, 8)
# Widen to +-500 nm / 50 nm: a Z4 = 0.10-wave defocus injects ~0.35 waves of
# paraxial OPD, i.e. needs dz ~ +-280 nm to be fully absorbed by the nuisance.
Z_OFFSET_GRID_NM: tuple[float, ...] = tuple(
    -500.0 + 50.0 * i for i in range(21))
ROI_RADIUS_UM: float = 1.5

# --- Experiment matrix (plan §6) -------------------------------------------------


@dataclass(frozen=True)
class Experiment:
    name: str
    truth: dict[int, float]
    peak: float
    scene_seed: int
    noise_seed: int
    meta_diameter_um: float | None = None   # None => scene value (correct anchor)
    note: str = ""


def experiment_matrix() -> list[Experiment]:
    """The five experiments of plan §6 (fixed seeds => reproducible report)."""
    return [
        Experiment("1_null", {}, 50.0, 1, 101,
                   note="zero injected; per-mode bias < 0.01 RMS waves"),
        Experiment("2_single_Z5", {5: 0.10}, 50.0, 1, 102,
                   note="single-mode 0.10; |err| < 0.02"),
        Experiment("2_single_Z6", {6: 0.10}, 50.0, 1, 103,
                   note="single-mode 0.10; |err| < 0.02"),
        Experiment("2_single_Z7", {7: 0.10}, 50.0, 1, 104,
                   note="single-mode 0.10; |err| < 0.02"),
        Experiment("2_single_Z8", {8: 0.10}, 50.0, 1, 105,
                   note="single-mode 0.10; |err| < 0.02"),
        Experiment("3_combined_snr20", dict(COMBINED_COEFFS), 20.0, 1, 106,
                   note="combined 4 modes @ SNR 20"),
        Experiment("3_combined_snr50", dict(COMBINED_COEFFS), 50.0, 1, 107,
                   note="combined 4 modes @ SNR 50; |err| < 0.03"),
        Experiment("3_combined_snr100", dict(COMBINED_COEFFS), 100.0, 1, 108,
                   note="combined 4 modes @ SNR 100; RMS ~ 1/sqrt(SNR): "
                        "SNR20->SNR100 expects ~2x lower RMS"),
        Experiment("4_defocus_leakage", {4: 0.10}, 50.0, 1, 109,
                   note="inject Z4 only; fitted-mode leakage < 0.02 "
                        "(absorbed by z-offset nuisance, needs |dz| ~ 280 nm)"),
        Experiment("5_wrong_anchor", dict(COMBINED_COEFFS), 50.0, 1, 110,
                   meta_diameter_um=0.080,
                   note="meta says 80 nm, truth 60 nm; quantify induced errors"),
    ]


# --- Real-data parameters (spec real-data-bacteria-psf.md §1, probe 2026-10-19)

#: lateral pixel size, µm (CZI Scaling X = Y = 1.135e-7 m)
REAL_PIXEL_UM: float = 0.1135
#: z-step, µm (CZI Scaling Z = 5e-7 m)
REAL_Z_STEP_UM: float = 0.5
#: AF488 emission wavelength, nm (CZI dye metadata)
REAL_LAMBDA_NM: float = 517.0
#: LD LCI Plan-Apochromat 40x/1.2 water objective
REAL_NA: float = 1.2
REAL_N_IM: float = 1.33
#: V. cholerae width anchor (spec §0: band 0.5–0.8 µm, nominal 0.65)
REAL_ANCHOR_UM: float = 0.65
REAL_ANCHOR_BAND_UM: tuple[float, float] = (0.5, 0.8)
#: fitted modes (spec §0 / plan: differential modes only)
REAL_MODES: tuple[int, ...] = (5, 6, 7, 8)
#: CZI channel index of AF488 (spec §1: "C3" in filenames ≠ CZI channel 3)
REAL_CHANNEL: int = 1
#: fluorescence channels -> emission λ, nm (spec §1)
REAL_CHANNEL_LAMBDA_NM: dict[int, float] = {
    0: 465.0,   # DAPI
    1: 517.0,   # AF488
    2: 568.0,   # AF555
    4: 636.0,   # mKate2
}
REAL_CZI_NAME: str = "GoC-Exp4_C3_7.czi"
REAL_CROP_NAME: str = "GoC-Exp4_C3_7-1_3d.tif"

#: per-segment z-candidate offsets, µm (spec §4, simplified 2026-10-19:
#: 0.5 µm steps match the data's z resolution — finer is unidentifiable)
Z_CANDIDATE_GRID_UM: tuple[float, ...] = (
    -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5)

#: re-export for the real path (single source of truth above)
ROI_RADIUS_UM: float = 1.5

#: bacteria-scale planar scene for the pre-flight (spec §5 Phase-2 parameters)
BACTERIA_SCENE: SceneSpec = SceneSpec(
    fov_px=256,
    pixel_size_um=REAL_PIXEL_UM,
    n_planes=15,
    z_step_um=REAL_Z_STEP_UM,
    n_segments=10,
    length_range_um=(1.0, 3.0),
    min_sep_um=1.5,
    border_margin_um=1.5,
    max_angular_gap_deg=100,
    fluorescent_diameter_um=REAL_ANCHOR_UM,
    supersample=8,
)


def real_pupil_config(fov_px: int, lambda_nm: float = REAL_LAMBDA_NM) -> PupilConfig:
    """PupilConfig at the measured real-data parameters (spec §1)."""
    return PupilConfig(
        pixel_size=REAL_PIXEL_UM,
        lambda_um=lambda_nm / 1000.0,
        na=REAL_NA,
        n_im=REAL_N_IM,
        grid_size=fov_px,
    )
