"""Export a 3D PSF volume as a plain multi-page TIFF for DOLPHIN.

DOLPHIN reads a PSF as one 3D TIF (dolphin_image::TiffReader, channel 0,
no OME assumptions), so we write a non-OME float32 multi-page stack.

Usage:
  # ideal (zero-Zernike) PSF, same 13x96x96 kernel the pipeline uses:
  .venv/bin/python export_psf_for_dolphin.py --out psf_ideal.tif

  # or a fitted PSF from the fit's psf.json:
  .venv/bin/python export_psf_for_dolphin.py --psf results/real_crop/psf.json --out psf_fitted.tif
"""
import argparse
import json

import numpy as np
import tifffile

from sibilla.defaults import REAL_LAMBDA_NM, REAL_NA, REAL_N_IM, REAL_PIXEL_UM
from sibilla.deconvolve import build_psf_volume
from sibilla.psf import PupilConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--psf", default=None,
                    help="psf.json from fit_real_crop (fitted PSF); if omitted, "
                         "writes the ideal (zero-Zernike) PSF")
    ap.add_argument("--z-step-um", type=float, default=0.5)
    args = ap.parse_args()

    if args.psf:
        meta = json.loads(open(args.psf).read())
        cfg = PupilConfig(pixel_size=float(meta["pixel_size_um"]),
                          lambda_um=float(meta["lambda_nm"]) / 1000.0,
                          na=float(meta["na"]), n_im=float(meta["n_im"]),
                          grid_size=96)
        coeffs = {int(k): float(v) for k, v in meta["coeffs"].items()}
        model = meta.get("psf_model", "widefield")
        lam_exc = float(meta.get("lambda_exc_nm", 493.0))
    else:
        cfg = PupilConfig(pixel_size=REAL_PIXEL_UM,
                          lambda_um=REAL_LAMBDA_NM / 1000.0,
                          na=REAL_NA, n_im=REAL_N_IM, grid_size=96)
        coeffs, model, lam_exc = {}, "widefield", 493.0

    vol = build_psf_volume(cfg, coeffs, args.z_step_um, shape=(13, 96, 96),
                           model=model, lambda_exc_nm=lam_exc)
    tifffile.imwrite(args.out, np.ascontiguousarray(vol, dtype=np.float32))
    print(f"wrote {args.out}: shape={vol.shape} (13 x 96 x 96), "
          f"model={model}, coeffs={coeffs}")


if __name__ == "__main__":
    main()
