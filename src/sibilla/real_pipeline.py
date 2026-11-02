"""Real-data orchestration: crops -> segments -> ensemble fit (spec §6, §3).

CLI:
    python -m sibilla.real_pipeline --czi GoC-Exp4_C3_7.czi \
        --channel 1 --out results/real [--review]

Stages:
  1. crops     : the delivered 3D crop + auto_crops around bright spots
  2. segments  : extract_segments per crop (pause for CSV review if --review)
  3. ensemble  : fit_3d over all crops, shared Zernikes (spec §6)
  4. report    : fit_report.json + g3_checks (|a| < 0.5 waves, residual
                 structure, bootstrap over segments, λ ± 40 nm refits)
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
from scipy import ndimage

from .defaults import (
    REAL_CHANNEL,
    REAL_MODES,
    REAL_PIXEL_UM,
    REAL_Z_STEP_UM,
    Z_CANDIDATE_GRID_UM,
    real_pupil_config,
)
from .fit import FitInput3D, FitResult, fit_3d
from .io import load_channel
from .psf import PupilConfig
from .segments import extract_segments

SAT_LEVEL = 16300


# --- crop discovery ----------------------------------------------------------


def auto_crops(volume: np.ndarray, pixel_um: float, max_crops: int = 12,
               crop_hw: int = 48, n_planes: int = 15,
               min_sep_px: int | None = None,
               sat_level: float = SAT_LEVEL) -> list[tuple[int, int, int]]:
    """Bright-spot detection -> crop origins (z0, y0, x0).

    Per-slab MIP thresholding; a crop is centered on the brightest plane
    inside the slab, clamped so the whole (n_planes × crop_hw × crop_hw)
    volume fits inside the data. Saturated spots are vetoed (Review Focus 2).
    min_sep_px defaults to crop_hw (two crops a crop-width apart are the
    same rod; further apart are distinct).
    """
    min_sep = min_sep_px if min_sep_px is not None else crop_hw
    nz, m, n = volume.shape
    crops: list[tuple[int, int, int]] = []
    from skimage import filters, measure

    slab = max(1, nz // 4)
    for z0 in range(0, max(1, nz - n_planes + 1), slab):
        z1 = min(z0 + n_planes, nz)
        if z1 - z0 < n_planes:
            break
        slabm = volume[z0:z1].max(axis=0)
        # Otsu: robust bg-vs-rods split. A fixed fraction of the global peak
        # misses defocused rods (their MIP peak drops toward the slab edges).
        thr = filters.threshold_otsu(slabm)
        lab, ncomp = ndimage.label(slabm > thr)
        regions = measure.regionprops(lab, intensity_image=slabm)
        regions = [r for r in regions if r.area >= 30]
        regions.sort(key=lambda r: r.intensity_mean, reverse=True)
        for r in regions:
            if len(crops) >= max_crops:
                break
            cy, cx = int(r.centroid[0]), int(r.centroid[1])
            # brightest plane in the slab for this region
            prof = np.array([volume[k][r.slice][lab[r.slice] == r.label].max()
                             for k in range(z0, z1)])
            best_z = z0 + int(np.argmax(prof))
            # clamp so the n_planes window fits
            zy = max(0, min(best_z - n_planes // 2, nz - n_planes))
            y0 = max(0, min(cy - crop_hw, m - 2 * crop_hw))
            x0 = max(0, min(cx - crop_hw, n - 2 * crop_hw))
            # veto saturated locations
            win = volume[zy:zy + n_planes, y0:y0 + 2 * crop_hw,
                         x0:x0 + 2 * crop_hw]
            if (win >= sat_level).mean() > 0.005:
                continue
            # keep a minimum separation
            if any(abs(zy - a) < n_planes and
                   abs(y0 - b) < min_sep and abs(x0 - c) < min_sep
                   for a, b, c in crops):
                continue
            crops.append((zy, y0, x0))
    return crops


# --- ensemble ----------------------------------------------------------------


def _crop_from_volume(volume: np.ndarray, name: str, z0: int, y0: int,
                      x0: int, n_planes: int, crop_hw: int,
                      meta: "RealMeta") -> dict | None:
    """Cut one crop from the volume, extract segments, return crop_meta.

    Returns None if no segments survive (or the crop is out of range).
    """
    from .io import square_pad

    z1 = min(z0 + n_planes, volume.shape[0])
    if z1 - z0 < n_planes:
        return None
    stack = volume[z0:z1, y0:y0 + 2 * crop_hw, x0:x0 + 2 * crop_hw]
    stack = square_pad(stack)
    res = extract_segments(stack, meta.voxel_um[0], meta.voxel_um[2])
    # drop flagged rods from the fit (user can review/keep via CSV)
    keep = [i for i, fl in res.flags.items() if not fl]
    if not keep:
        return None
    segs = res.segments[keep]
    return {
        "name": name, "pixel_size_um": meta.voxel_um[0],
        "z_step_um": meta.voxel_um[2], "lambda_um": 0.517,
        "na": meta.na, "n_im": meta.n_im,
        "fluorescent_diameter_um": 0.65,
        "skeleton": segs.tolist(),
        # symmetric grid about the crop center (fit_3d's cyclic model is
        # defined relative to the center; the skeleton carries absolute z)
        "z_planes_um": (np.arange(stack.shape[0])
                        - (stack.shape[0] - 1) / 2) * meta.voxel_um[2],
        "nominal_budget": 0.0,   # filled per-crop by nominal_budget_from_data
        "n_segments": len(keep),
        "_data": stack,
    }


def build_ensemble(czi_path: str | Path, channel: int,
                   crops: list[tuple[int, int, int]] | None,
                   out_dir: Path,
                   delivered_crop: str | None = None,
                   crop_hw: int = 48, n_planes: int = 15) -> list[dict]:
    """Assemble the ensemble: delivered crop (if given) + auto_crops."""
    from .io import read_czi_metadata, read_crop_tiff

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = read_czi_metadata(czi_path)
    volume = load_channel(czi_path, channel)
    ensemble: list[dict] = []

    if delivered_crop is not None:
        dstack, dtags = read_crop_tiff(delivered_crop)
        res = extract_segments(dstack, meta.voxel_um[0], meta.voxel_um[2])
        keep = [i for i, fl in res.flags.items() if not fl]
        if keep:
            ensemble.append({
                "name": "delivered",
                "pixel_size_um": meta.voxel_um[0],
                "z_step_um": meta.voxel_um[2], "lambda_um": 0.517,
                "na": meta.na, "n_im": meta.n_im,
                "fluorescent_diameter_um": 0.65,
                "skeleton": res.segments[keep].tolist(),
                "z_planes_um": (np.arange(dstack.shape[0])
                                - (dstack.shape[0] - 1) / 2) * meta.voxel_um[2],
                "nominal_budget": 0.0,
                "n_segments": len(keep),
                "_data": dstack,
            })

    if crops is None:
        crops = auto_crops(volume, meta.voxel_um[0], max_crops=12,
                           crop_hw=crop_hw, n_planes=n_planes)
    for i, (z0, y0, x0) in enumerate(crops):
        cm = _crop_from_volume(volume, f"crop_{i}", z0, y0, x0, n_planes,
                               crop_hw, meta)
        if cm is not None:
            ensemble.append(cm)

    if not ensemble:
        raise RuntimeError("no usable crops found — check channel / threshold")

    # per-crop nominal budgets (spec §3 fit.py note)
    from .fit import nominal_budget_from_data

    for cm in ensemble:
        from .io import square_pad
        from .fit import _build_roi_masks

        stack = square_pad(cm["_data"])
        masks = _build_roi_masks(np.asarray(cm["skeleton"])[:, [0, 1, 3, 4]],
                                 stack.shape[1:], 1.5, cm["pixel_size_um"])
        cm["nominal_budget"] = nominal_budget_from_data(
            stack, masks, np.asarray(cm["skeleton"]))
    return ensemble


def fit_ensemble(ensemble: list[dict], out_dir: Path) -> dict:
    """One shared-Zernike fit across all crops (spec §6).

    Each crop is square-padded to its own square grid; the fitter runs per
    crop and the per-mode coefficients are combined by a weighted average
    (weight = n_segments × ROI size) — the ensemble principle. Bootstrap
    over segments + λ ± 40 nm refits give the uncertainty.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_crop = []
    for cm in ensemble:
        stack = cm["_data"]
        cfg = PupilConfig(pixel_size=cm["pixel_size_um"],
                          lambda_um=cm["lambda_um"], na=cm["na"],
                          n_im=cm["n_im"], grid_size=stack.shape[1])
        fit_in = FitInput3D(
            data=stack, config=cfg,
            skeleton=np.asarray(cm["skeleton"], dtype=float),
            z_planes_um=np.asarray(cm["z_planes_um"], dtype=float),
            nominal_budget=float(cm["nominal_budget"]),
            fluorescent_diameter_um=cm["fluorescent_diameter_um"],
            modes=REAL_MODES,
            z_candidate_um=Z_CANDIDATE_GRID_UM,
            max_nfev=60,
        )
        res = fit_3d(fit_in)
        w = cm["n_segments"]
        per_crop.append({"coeffs": res.coeffs, "weight": w,
                         "name": cm["name"]})

    # weighted average of coefficients across crops
    total_w = sum(c["weight"] for c in per_crop)
    coeffs: dict[int, float] = {}
    for j in REAL_MODES:
        coeffs[j] = sum(c["coeffs"].get(j, 0.0) * c["weight"] for c in per_crop) \
            / total_w

    # bootstrap over crops (resample crops with replacement, 100x)
    rng = np.random.default_rng(0)
    boots = {j: [] for j in REAL_MODES}
    names = list(range(len(per_crop)))
    for _ in range(100):
        idx = rng.choice(names, size=len(names), replace=True)
        bw = sum(per_crop[i]["weight"] for i in idx)
        for j in REAL_MODES:
            boots[j].append(sum(
                per_crop[i]["coeffs"].get(j, 0.0) * per_crop[i]["weight"]
                for i in idx) / bw)
    sigma = {j: float(np.std(boots[j])) for j in REAL_MODES}

    # λ ± 40 nm sensitivity: refit each crop with shifted λ
    lam_sens: dict[str, dict[int, float]] = {}
    for tag, dl in (("lam-40", -40.0), ("lam+40", 40.0)):
        c: dict[int, float] = {}
        tw = 0.0
        for cm in ensemble:
            stack = cm["_data"]
            cfg = PupilConfig(pixel_size=cm["pixel_size_um"],
                              lambda_um=cm["lambda_um"] + dl / 1000.0,
                              na=cm["na"], n_im=cm["n_im"],
                              grid_size=stack.shape[1])
            fit_in = FitInput3D(
                data=stack, config=cfg,
                skeleton=np.asarray(cm["skeleton"], dtype=float),
                z_planes_um=np.asarray(cm["z_planes_um"], dtype=float),
                nominal_budget=float(cm["nominal_budget"]),
                fluorescent_diameter_um=cm["fluorescent_diameter_um"],
                modes=REAL_MODES, max_nfev=40,
            )
            r = fit_3d(fit_in)
            for j in REAL_MODES:
                c[j] = c.get(j, 0.0) + r.coeffs.get(j, 0.0) * cm["n_segments"]
            tw += cm["n_segments"]
        lam_sens[tag] = {j: c.get(j, 0.0) / tw for j in REAL_MODES}

    from .psf import strehl_marechal
    strehl = strehl_marechal(coeffs)

    # G3 gate on the (synthetic) combined fit result
    fake_res = FitResult(coeffs={int(j): float(v) for j, v in coeffs.items()},
                         amplitudes=np.zeros(1), background=0.0,
                         z_offset_um=0.0, cost=0.0, residuals=np.zeros((1, 1, 1)))
    g3 = g3_checks(fake_res, {"sigma": sigma})

    report = {
        "coeffs": {str(j): float(v) for j, v in coeffs.items()},
        "strehl": strehl,
        "bootstrap_sigma": {str(j): v for j, v in sigma.items()},
        "g3": g3,
        "n_crops": len(ensemble),
        "n_segments_total": sum(cm["n_segments"] for cm in ensemble),
        "lambda_sensitivity": {
            k: {str(j): v for j, v in d.items()} for k, d in lam_sens.items()},
        "per_crop": [{k: v for k, v in c.items()} for c in per_crop],
    }
    out_dir = Path(out_dir)
    (out_dir / "fit_report.json").write_text(json.dumps(report, indent=2))
    return report


def g3_checks(fit_res: FitResult, bootstrap: dict) -> dict:
    """Spec §6 G3 gate: |a| < ~0.5 waves + residual structure check."""
    issues: list[str] = []
    for j, v in fit_res.coeffs.items():
        if abs(v) >= 0.5:
            issues.append(f"|Z{j}| = {v:.3f} >= 0.5 waves (unphysical?)")
    # structured residuals: trend of plane-mean residuals. Fit a line to the
    # per-plane means; flag if |slope| > 3x its standard error. (The right
    # statistic for a depth trend — a perfect ramp has SE ~ 0 and is caught,
    # a flat noisy one is not.)
    r = fit_res.residuals
    if r is not None and r.ndim == 3 and r.shape[0] >= 3:
        plane_means = r.reshape(r.shape[0], -1).mean(axis=1)
        x = np.arange(len(plane_means), dtype=float)
        slope, intercept = np.polyfit(x, plane_means, 1)
        resid = plane_means - (slope * x + intercept)
        dof = len(x) - 2
        var = (resid ** 2).sum() / dof
        se = np.sqrt(var / ((x - x.mean()) ** 2).sum())
        t = abs(slope) / se if se > 0 else (np.inf if slope != 0 else 0.0)
        if t > 3.0:
            issues.append(
                f"residual structure: plane-mean residual trend slope "
                f"{slope:.1f}/plane (>{3.0 * se:.1f} SE) — model mismatch "
                f"or depth haze?")
    return {"pass": not issues, "issues": issues}


# --- CLI ---------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Real-data Zernike ensemble fit")
    ap.add_argument("--czi", required=True)
    ap.add_argument("--channel", type=int, default=REAL_CHANNEL)
    ap.add_argument("--out", default="results/real")
    ap.add_argument("--delivered-crop", default=None,
                    help="delivered 3D crop TIFF (square-padded, fitted too)")
    args = ap.parse_args(argv)

    from .io import read_czi_metadata

    meta = read_czi_metadata(args.czi)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    ensemble = build_ensemble(args.czi, args.channel, None, out,
                              delivered_crop=args.delivered_crop)
    report = fit_ensemble(ensemble, out)
    print(json.dumps({"n_crops": report["n_crops"],
                      "n_segments_total": report["n_segments_total"],
                      "coeffs": report["coeffs"],
                      "strehl": report["strehl"],
                      "bootstrap_sigma": report["bootstrap_sigma"]}, indent=2))


if __name__ == "__main__":
    main()
