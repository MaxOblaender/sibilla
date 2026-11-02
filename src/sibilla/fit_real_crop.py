"""Fit the delivered 3D crop directly (G2 relaxed to a robustness gate).

Usage (server):
    .venv/bin/python -m sibilla.fit_real_crop --crop GoC-Exp4_C3_7-1_3d.tif \
        --out results/real_crop
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .defaults import REAL_MODES, real_pupil_config
from .fit import FitInput3D, _build_roi_masks, fit_3d, nominal_budget_from_data
from .io import read_crop_tiff, square_pad
from .psf import strehl_marechal
from .segments import extract_segments, make_review_figure


def fit_crop(crop_path: str | Path, out_dir: str | Path,
             keep_flagged: bool = False,
             diameter_um: float = 0.65,
             max_nfev: int = 60,
             z_range_um: float = 1.5,
             psf_model: str = "widefield",
             lambda_exc_nm: float = 493.0,
             use_segments: list[int] | None = None) -> dict:
    """Fit Zernikes to one 3D crop; write fit_report.json + review figure."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stack, tags = read_crop_tiff(crop_path)
    pixel_um = tags.get("pixel_size_um", 0.1135)
    z_step = tags.get("z_step_um", 0.5)
    padded = square_pad(stack)
    nz, m, n = padded.shape

    seg_res = extract_segments(padded, pixel_um, z_step)
    if use_segments is not None:
        keep = [int(i) for i in use_segments]
        if any(i < 0 or i >= len(seg_res.segments) for i in keep):
            raise RuntimeError(
                f"segment index out of range: {keep} (have "
                f"{len(seg_res.segments)})")
    else:
        keep = (list(range(len(seg_res.segments))) if keep_flagged
                else [i for i, fl in seg_res.flags.items() if not fl])
    if not keep:
        raise RuntimeError("no unflagged segments found in crop")
    skel = seg_res.segments[keep]
    make_review_figure(padded, seg_res, out_dir / "segments_review.png")

    masks = _build_roi_masks(skel[:, [0, 1, 3, 4]], (m, n), 1.5, pixel_um)
    bnom = nominal_budget_from_data(padded, masks, skel)

    cfg = real_pupil_config(n)
    z_planes = (np.arange(nz) - (nz - 1) / 2) * z_step
    # z-candidate grid: ±z_range_um in z_step increments (default 1.5 µm)
    z_cands = tuple(
        -z_range_um + i * z_step for i in range(int(round(2 * z_range_um / z_step)) + 1))
    fit_in = FitInput3D(
        data=padded, config=cfg, skeleton=skel, z_planes_um=z_planes,
        nominal_budget=bnom, fluorescent_diameter_um=diameter_um,
        modes=REAL_MODES, z_candidate_um=z_cands,
        psf_model=psf_model, lambda_exc_nm=lambda_exc_nm,
        max_nfev=max_nfev,
    )
    res = fit_3d(fit_in)

    report = {
        "crop": str(crop_path),
        "psf_model": psf_model,
        "z_candidate_range_um": [float(z_cands[0]), float(z_cands[-1])],
        "z_pinned_at_boundary": bool(
            any(abs(z) >= abs(z_cands[-1]) - 1e-9
                for z in res.segment_z_um)),
        "n_segments_extracted": int(len(seg_res.segments)),
        "n_segments_fitted": int(len(keep)),
        "flags": {str(i): fl for i, fl in seg_res.flags.items()},
        "coeffs": {str(j): float(v) for j, v in res.coeffs.items()},
        "strehl": strehl_marechal(res.coeffs),
        "segment_z_um": [float(v) for v in res.segment_z_um],
        "amplitudes": [float(v) for v in res.amplitudes],
        "background_counts": float(res.background),
        "cost": float(res.cost),
        "assumptions": [
            "widefield scalar PSF model on spinning-disk data (accepted risk)",
            "fixed specimen (unverified)",
            "anchor diameter 0.65 um (band 0.5-0.8)",
        ],
    }
    (out_dir / "fit_report.json").write_text(json.dumps(report, indent=2))
    # canonical PSF spec for the deconvolve CLI (per-channel λ at read time)
    (out_dir / "psf.json").write_text(json.dumps({
        "coeffs": {str(j): float(v) for j, v in res.coeffs.items()},
        "na": cfg.na, "n_im": cfg.n_im,
        "lambda_nm": cfg.lambda_um * 1000.0,
        "pixel_size_um": cfg.pixel_size,
        "z_step_um": float(z_step),
        "psf_model": psf_model,
        "lambda_exc_nm": lambda_exc_nm,
    }, indent=2))
    return report


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Fit Zernikes to one 3D crop")
    ap.add_argument("--crop", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--keep-flagged", action="store_true",
                    help="fit flagged segments too (default: unflagged only)")
    ap.add_argument("--use-segments", type=str, default=None,
                    help="comma-separated segment INDICES to fit (overrides "
                         "flag logic; from manual_crop.py metrics/segments.csv)")
    ap.add_argument("--diameter-um", type=float, default=0.65)
    ap.add_argument("--max-nfev", type=int, default=60)
    ap.add_argument("--z-range-um", type=float, default=1.5,
                    help="per-segment z-candidate range ± (default 1.5 µm)")
    ap.add_argument("--psf-model", choices=["widefield", "confocal"],
                    default="widefield",
                    help="PSF model: widefield (single path) or confocal "
                         "(spinning-disk product, exc × det)")
    ap.add_argument("--lambda-exc-nm", type=float, default=493.0,
                    help="excitation wavelength nm (confocal model)")
    args = ap.parse_args(argv)
    rep = fit_crop(args.crop, args.out, args.keep_flagged,
                   args.diameter_um, args.max_nfev, args.z_range_um,
                   args.psf_model, args.lambda_exc_nm,
                   use_segments=([s for s in
                                  (int(t) for t in
                                   args.use_segments.split(",") if t != "")]
                                 if args.use_segments else None))
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
