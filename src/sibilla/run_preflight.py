"""Bacteria-scale synthetic pre-flight (spec §5, simplified 2026-10-19).

Four experiments (user decision: keep it simple — the anchor/λ sensitivity
studies P3/P4 and the separate depth-scatter P7 were folded in or dropped):

- P1_null             zero injection — bias floor
- P2_combined_depth   4 modes at real SNR WITH per-rod depth scatter
                      (the realistic case; absorbs old P7)
- P5_segment_count    rods = 1/3/6/10/20 — how many bacteria to crop
- P6_rl_roundtrip     3D RL with the fitted PSF — validates the width
                      re-measurement self-check (spec §6)

Gate G2: all |err| <= ~0.05 RMS waves at real SNR with a practical rod
count, else restrict the mode set or stop (spec §5).

Usage:
    python -m sibilla.run_preflight --out results/preflight [--only ...]
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from .defaults import (
    BACTERIA_SCENE,
    REAL_ANCHOR_UM,
    REAL_MODES,
    real_pupil_config,
)
from .fit import FitInput3D, fit_3d
from .forward import cyclic3d, psf3d_cyclic
from .psf import build_psf_stack
from .scene import CapsuleSpec, generate_capsule_scene, render_capsule_unit


@dataclass
class Preflight:
    name: str
    truth: dict[int, float]
    peak: float
    scene_seed: int
    noise_seed: int
    # P2: per-rod z scatter (µm, uniform draw width); P5: rod counts
    z_scatter_um: float = 0.0
    variants: tuple | None = None
    spec: object = None      # allows tests to shrink the scene


def preflight_experiments() -> list[Preflight]:
    return [
        Preflight("P1_null", {}, 4500.0, 1, 101),
        Preflight("P2_combined_depth", {5: 0.10, 6: 0.10, 7: 0.07, 8: 0.07},
                  4500.0, 2, 102, z_scatter_um=1.0),
        Preflight("P5_segment_count", {6: 0.10}, 4500.0, 3, 103,
                  variants=(1, 3, 6, 10, 20)),
        Preflight("P6_rl_roundtrip", {6: 0.10}, 4500.0, 4, 104,
                  z_scatter_um=1.0),
    ]


def _capsule_spec(fov: int, n_planes: int = 15) -> CapsuleSpec:
    return CapsuleSpec(fov_px_y=fov, fov_px_x=fov, n_planes=n_planes,
                       pixel_size_um=BACTERIA_SCENE.pixel_size_um,
                       z_step_um=BACTERIA_SCENE.z_step_um,
                       diameter_um=REAL_ANCHOR_UM, aa_width_px=1.0)


def make_capsule_data(spec: CapsuleSpec, coeffs: dict[int, float],
                      z_scatter_um: float, peak: float, bg: float,
                      seed: int, n_segments: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """Capsule scene -> noisy (Nz, M, N) counts + skeleton.

    Physically-correct depth model: each rod is placed at its scattered z
    center and blurred with the PSF evaluated at ITS OWN depth (cyclic-z
    3D convolution). `z_scatter_um` is the uniform half-width of the z
    scatter (µm around the mid-plane).
    """
    scene = generate_capsule_scene(
        spec, seed=seed, n_segments=n_segments,
        length_range_um=BACTERIA_SCENE.length_range_um,
        z_center_range_um=(-max(z_scatter_um, 1e-3), max(z_scatter_um, 1e-3)),
        max_tilt_deg=0.0)
    rng = np.random.default_rng(seed + 1000)
    cfg = real_pupil_config(spec.fov_px_x)
    psf = psf3d_cyclic(cfg, coeffs, spec.n_planes, spec.z_step_um)
    data = np.full((spec.n_planes, spec.fov_px_y, spec.fov_px_x), bg)
    for obj in scene.objects:
        omax = max(float(obj.max()), 1e-12)
        data += peak * cyclic3d(obj / omax, psf)
    return rng.poisson(data).astype(np.float64), scene.segments


def _n_segments_for(spec: CapsuleSpec) -> int:
    """Rod count scaled to the FOV (full 256-px scene -> 10, spec §5)."""
    area = (spec.fov_px_x * BACTERIA_SCENE.pixel_size_um) ** 2
    full = (BACTERIA_SCENE.fov_px * BACTERIA_SCENE.pixel_size_um) ** 2
    return max(1, int(round(BACTERIA_SCENE.n_segments * area / full)))


def run_one(exp: Preflight, out_dir: Path) -> dict:
    spec = exp.spec or _capsule_spec(BACTERIA_SCENE.fov_px)
    n_planes = spec.n_planes
    cfg = real_pupil_config(spec.fov_px_x)
    z_planes = (np.arange(n_planes) - (n_planes - 1) / 2) * spec.z_step_um
    rng = np.random.default_rng(exp.scene_seed + 5000)

    rec: dict = {"name": exp.name, "truth": exp.truth,
                 "per_mode_err": {}, "segments_used": 0, "variants": {}}

    if exp.name == "P5_segment_count":
        for n_seg in exp.variants:
            width = exp.z_scatter_um if exp.z_scatter_um > 0 else 1.0
            data, skel = make_capsule_data(
                spec, exp.truth, width, exp.peak, 250.0,
                exp.scene_seed * 31 + n_seg, n_segments=n_seg)
            fit_in = FitInput3D(
                data=data, config=cfg, skeleton=skel,
                z_planes_um=z_planes, nominal_budget=exp.peak,
                fluorescent_diameter_um=REAL_ANCHOR_UM, max_nfev=60)
            res = fit_3d(fit_in)
            errs = {j: abs(res.coeffs.get(j, 0.0) - exp.truth.get(j, 0.0))
                    for j in REAL_MODES}
            rec["variants"][str(n_seg)] = {"per_mode_err": errs}
            rec["per_mode_err"] = errs
            rec["segments_used"] = n_seg
    else:
        n_seg = _n_segments_for(spec)
        width = exp.z_scatter_um
        data, skel = make_capsule_data(
            spec, exp.truth, width, exp.peak, 250.0, exp.noise_seed,
            n_segments=n_seg)
        fit_in = FitInput3D(
            data=data, config=cfg, skeleton=skel,
            z_planes_um=z_planes, nominal_budget=exp.peak,
            fluorescent_diameter_um=REAL_ANCHOR_UM, max_nfev=60)
        res = fit_3d(fit_in)
        rec["per_mode_err"] = {
            j: abs(res.coeffs.get(j, 0.0) - exp.truth.get(j, 0.0))
            for j in REAL_MODES}
        rec["segments_used"] = n_seg
        rec["fitted"] = dict(res.coeffs)
        if exp.name == "P6_rl_roundtrip":
            # width self-check on the rod's OWN in-focus plane: data plane k
            # = round(seg_z) (defocus k - seg_z = 0), deconvolved with the
            # in-focus PSF. Background is subtracted first (RL with a
            # nonzero floor diverges).
            from .metrics import apparent_diameter_um, richardson_lucy

            seg = skel[0]
            kz = int(round(float(seg[2])))
            kz = min(max(kz, 0), n_planes - 1)
            plane = np.maximum(data[kz] - np.median(data[kz]), 0.0)
            psf_infocus = psf3d_cyclic(cfg, res.coeffs, n_planes,
                                       spec.z_step_um)[0]
            dec = richardson_lucy(plane, psf_infocus, n_iter=30)
            d_raw = apparent_diameter_um(
                plane, spec.pixel_size_um,
                center=(0.5 * (seg[0] + seg[3]), 0.5 * (seg[1] + seg[4])),
                direction=(seg[3] - seg[0], seg[4] - seg[1]))
            d = apparent_diameter_um(
                dec, spec.pixel_size_um,
                center=(0.5 * (seg[0] + seg[3]), 0.5 * (seg[1] + seg[4])),
                direction=(seg[3] - seg[0], seg[4] - seg[1]))
            rec["width_after_um"] = float(d)
            rec["width_raw_um"] = float(d_raw)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "result.json").write_text(json.dumps(rec, indent=2))
    return rec


def g2_verdict(results: dict) -> dict:
    """Pass if all |err| <= 0.05 waves on the combined-depth experiment."""
    key = "P2_combined_depth"
    errs = (results.get(key, {}).get("per_mode_err")
            if key in results else None)
    if not errs:
        return {"pass": False, "max_err_per_mode": {},
                "notes": f"{key} not run"}
    return {"pass": all(v <= 0.05 for v in errs.values()),
            "max_err_per_mode": errs,
            "notes": "gate: |err| <= 0.05 RMS waves at real SNR"}


def run_preflight(out_root: str | Path,
                  only: list[str] | None = None) -> dict:
    out_root = Path(out_root)
    results: dict[str, dict] = {}
    for exp in preflight_experiments():
        if only and exp.name not in only:
            continue
        results[exp.name] = run_one(exp, out_root / exp.name)
    results["_g2"] = g2_verdict(results)
    (out_root / "summary.json").write_text(json.dumps(results, indent=2))
    return results


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Bacteria-scale pre-flight")
    ap.add_argument("--out", default="results/preflight")
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args(argv)
    results = run_preflight(args.out, args.only)
    print(json.dumps(results["_g2"], indent=2))


if __name__ == "__main__":
    main()
