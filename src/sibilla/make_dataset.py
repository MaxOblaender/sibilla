"""CLI: generate one synthetic dataset (plan §4 data contract).

Outputs a directory containing:

- stack.tif   uint16 multi-page (low→high z), photon units
- meta.json   pixel size, z-step, λ, NA, n, fluorescent diameter,
              per-segment skeleton (endpoints, z), photon scale, modes to fit,
              z-plane defocus list, nominal per-tubule photon budget B_nom
- truth.json  injected coefficients, scene seed, noise seed, true z-offset
              (evaluation only — never read by the fitter)

Usage:
    python -m sibilla.make_dataset --out results/smoke \
        --scene-seed 0 --noise-seed 1 [--peak 50] [--coeff 5=0.10 6=0.05]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tifffile

from .defaults import (
    DEFAULT_SCENE,
    FITTED_MODES,
    LAMBDA_NM,
    NA,
    N_IM,
    background,
    nominal_budget,
    pupil_config,
)
from .forward import forward_stack, render_uint16
from .psf import build_psf_stack
from .scene import generate_scene, render_scene_unit


def z_planes(spec) -> np.ndarray:
    """Detector plane defocus (µm), plane 0 = object plane, symmetric."""
    nz = spec.n_planes
    return (np.arange(nz) - (nz - 1) / 2.0) * spec.z_step_um


def parse_coeff_tokens(tokens: list[str] | None) -> dict[int, float]:
    """Parse ['5=0.10', '6=-0.05'] -> {5: 0.10, 6: -0.05}."""
    out: dict[int, float] = {}
    for t in tokens or []:
        j, v = t.split("=")
        out[int(j)] = float(v)
    return out


def generate_dataset(
    out_dir: str | Path,
    scene_seed: int = 0,
    noise_seed: int = 0,
    truth_coeffs: dict[int, float] | None = None,
    peak: float = 50.0,
    spec=None,
    true_z_offset_um: float = 0.0,
    meta_diameter_um: float | None = None,
    write_files: bool = True,
) -> dict:
    """Generate the full dataset; return a summary dict.

    truth_coeffs : injected Zernike coefficients (RMS waves), any Noll set.
                   Modes outside FITTED_MODES (e.g. Z4) are still injected
                   into the PSF — they are just not fitted (experiment 4).
    meta_diameter_um : if given, meta.json carries this (WRONG) diameter
                   while the scene is rendered at the true one (experiment 5).
    """
    spec = spec or DEFAULT_SCENE
    truth_coeffs = dict(truth_coeffs or {})
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scene = generate_scene(spec, seed=scene_seed)
    truth_diameter = spec.fluorescent_diameter_um
    objects = render_scene_unit(scene, diameter_um=truth_diameter)

    cfg = pupil_config(spec)
    zp = z_planes(spec) + true_z_offset_um          # object offset from detectors
    psf = build_psf_stack(cfg, truth_coeffs, zp)    # PSF at each detector plane

    B_nom = nominal_budget(peak, spec)
    bg = background(peak)
    rng = np.random.default_rng(noise_seed)

    # forward: sum of blurred unit objects (each at amp=1, budget B_nom) + bg
    o_tot = np.sum(objects, axis=0)                 # (M, N), photons at amp 1
    data = forward_stack(o_tot, psf, amplitude=B_nom, background=bg,
                         noise=True, rng=rng)
    u16 = render_uint16(data)

    skeleton = scene.segments.copy()
    skeleton_z = np.zeros(skeleton.shape[0])        # object plane (given z = 0)

    meta = {
        "pixel_size_um": spec.pixel_size_um,
        "z_step_um": spec.z_step_um,
        "n_planes": spec.n_planes,
        "stack_shape": [spec.n_planes, spec.fov_px, spec.fov_px],
        "z_planes_um": (zp - true_z_offset_um).tolist(),   # detector grid (no true offset)
        "lambda_nm": LAMBDA_NM,
        "lambda_um": LAMBDA_NM / 1000.0,
        "na": NA,
        "n_im": N_IM,
        "fluorescent_diameter_um": (meta_diameter_um if meta_diameter_um is not None
                                    else truth_diameter),
        "skeleton": skeleton.tolist(),
        "skeleton_z_um": skeleton_z.tolist(),
        "photon_scale": 1.0,
        "nominal_budget": B_nom,
        "background_expected_fraction": 0.05,
        "peak": peak,
        "modes": list(FITTED_MODES),
        "conventions": {
            "basis": "Noll, RMS-normalized over unit pupil",
            "coeff_units": "RMS waves",
            "psf": "scalar widefield, energy-normalized per plane",
            "scene": "planar object; image_j = O conv PSF(z_j) + bg (+Poisson)",
            "space_units": "um (diameter in um here; input spec in nm)",
        },
    }
    truth = {
        "coeffs": truth_coeffs,
        "scene_seed": scene_seed,
        "noise_seed": noise_seed,
        "true_z_offset_um": true_z_offset_um,
        "truth_diameter_um": truth_diameter,
        "peak": peak,
        "background": bg,
        "nominal_budget": B_nom,
        "n_segments": scene.n_segments,
        "segment_lengths_um": scene.length_um.tolist(),
    }

    if write_files:
        tifffile.imwrite(out_dir / "stack.tif", u16,
                         metadata={"pixel_size_um": spec.pixel_size_um,
                                   "z_step_um": spec.z_step_um})
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
        (out_dir / "truth.json").write_text(json.dumps(truth, indent=2))

    return {
        "out_dir": str(out_dir),
        "n_segments": scene.n_segments,
        "truth": truth,
        "meta": meta,
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Generate one synthetic Zernike dataset")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--scene-seed", type=int, default=0)
    ap.add_argument("--noise-seed", type=int, default=0)
    ap.add_argument("--peak", type=float, default=50.0, help="peak photons/px")
    ap.add_argument("--coeff", action="append", default=None,
                    help="injected coefficient 'Z=amp' (RMS waves), repeatable")
    ap.add_argument("--z-offset", type=float, default=0.0, help="true z-offset, µm")
    ap.add_argument("--meta-diameter-nm", type=float, default=None,
                    help="override the diameter written to meta.json (nm)")
    args = ap.parse_args(argv)
    meta_d = (args.meta_diameter_nm / 1000.0) if args.meta_diameter_nm else None
    summary = generate_dataset(
        args.out,
        scene_seed=args.scene_seed,
        noise_seed=args.noise_seed,
        truth_coeffs=parse_coeff_tokens(args.coeff),
        peak=args.peak,
        true_z_offset_um=args.z_offset,
        meta_diameter_um=meta_d,
    )
    print(json.dumps({"out_dir": summary["out_dir"],
                      "n_segments": summary["n_segments"],
                      "truth_coeffs": summary["truth"]["coeffs"]}, indent=2))


if __name__ == "__main__":
    main()
