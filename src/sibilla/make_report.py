"""Report generator: results/experiments.json -> results/report/report.md + figures.

Figures (one per report):
- fig_coeffs.png      fitted vs injected coefficients per experiment (grouped bars)
- fig_combined_snr.png wavefront RMS error vs SNR (experiment 3)
- fig_stacks_<exp>.png in-focus slice: raw vs forward model vs residuals
- fig_psf_<exp>.png   in-focus PSF (fitted) vs true PSF (peak-normalized)

Usage:
    python -m sibilla.make_report --results results
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tifffile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .defaults import DEFAULT_SCENE
from .fit import fit_from_files
from .zernike import mode_name

MODES = (5, 6, 7, 8)


def _load_stack(ds: str) -> tuple[np.ndarray, dict, dict]:
    stack = np.asarray(tifffile.imread(Path(ds) / "stack.tif"), float)
    meta = json.loads((Path(ds) / "meta.json").read_text())
    truth = json.loads((Path(ds) / "truth.json").read_text())
    return stack, meta, truth


def _fig_coeffs(exps: dict, out: Path) -> None:
    names = list(exps.keys())
    x = np.arange(len(MODES))
    w = 0.8 / len(names)
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, nm in enumerate(names):
        errs = [exps[nm]["per_mode_err"].get(str(j), 0.0) for j in MODES]
        ax.bar(x + (i - (len(names) - 1) / 2) * w, errs, width=w,
               label=f"{nm} ({exps[nm]['truth_coeffs'] and 'injected ' + ', '.join(f'Z{k}={v:.2f}' for k, v in exps[nm]['truth_coeffs'].items() if float(v)) or 'null'})")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([mode_name(j) for j in MODES])
    ax.set_ylabel("fitted − injected (RMS waves)")
    ax.set_title("Coefficient error per experiment")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def _fig_snr(exps: dict, out: Path) -> None:
    sns = []
    rms = []
    for nm in ("3_combined_snr20", "3_combined_snr50", "3_combined_snr100"):
        if nm in exps:
            sns.append(int(nm.split("snr")[1]))
            rms.append(exps[nm]["wavefront_rms_error"])
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(sns, rms, "o-")
    ax.set_xlabel("peak SNR (photons/px)")
    ax.set_ylabel("wavefront RMS error (waves)")
    ax.set_title("Experiment 3: combined 4-mode recovery vs SNR")
    for s, r in zip(sns, rms):
        ax.annotate(f"{r:.4f}", (s, r), textcoords="offset points", xytext=(5, 5))
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def _fitted_forward_plane(exp: dict, plane: int) -> np.ndarray:
    """Forward model of one plane with the stored fitted solution
    (per-segment amplitudes respected)."""
    stack, meta, truth = _load_stack(exp["dataset"])
    fitted = {int(k): v for k, v in exp["fitted_coeffs"].items()}
    z_offset = exp["z_offset_um"]
    amps = np.asarray(exp["amplitudes"], float)
    from .psf import build_psf_stack
    from .scene import render_scene_unit, generate_scene
    from .defaults import pupil_config
    from .forward import FastBlur
    scene = generate_scene(DEFAULT_SCENE, seed=truth["scene_seed"])
    objs = render_scene_unit(scene, diameter_um=meta["fluorescent_diameter_um"])
    cfg = pupil_config(DEFAULT_SCENE)
    zp = np.array(meta["z_planes_um"]) + z_offset
    psf = build_psf_stack(cfg, fitted, zp)
    pred = np.zeros((meta["stack_shape"][1], meta["stack_shape"][2]))
    for s, o in enumerate(objs):
        fb = FastBlur(o)
        pred += amps[s] * meta["nominal_budget"] * fb.blur(psf.psf[plane])
    pred += exp["background_fitted"]
    return pred, stack, meta


def _fig_stack(exp_name: str, exp: dict, out: Path) -> None:
    _, meta, _ = _load_stack(exp["dataset"])
    k = int(np.argmin(np.abs(np.asarray(meta["z_planes_um"]) + exp["z_offset_um"])))
    pred, stack, meta = _fitted_forward_plane(exp, k)
    res = stack[k] - pred
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    vmax = np.percentile(stack, 99)
    im0 = ax[0].imshow(stack[k], cmap="gray", vmin=0, vmax=vmax)
    ax[0].set_title(f"raw plane {k}")
    ax[1].imshow(pred, cmap="gray", vmin=0, vmax=vmax)
    ax[1].set_title("fitted forward model")
    im2 = ax[2].imshow(res, cmap="RdBu_r",
                       vmin=-max(1.0, np.percentile(np.abs(res), 99)),
                       vmax=max(1.0, np.percentile(np.abs(res), 99)))
    ax[2].set_title("residual")
    for a in ax:
        a.axis("off")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def _fig_psf(exp_name: str, exp: dict, out: Path) -> None:
    _, meta, truth = _load_stack(exp["dataset"])
    from .defaults import pupil_config
    from .psf import build_psf_stack
    cfg = pupil_config(DEFAULT_SCENE)
    k = int(np.argmin(np.abs(np.asarray(meta["z_planes_um"]) + exp["z_offset_um"])))
    fitted = {int(j): v for j, v in exp["fitted_coeffs"].items()}
    true_psf = build_psf_stack(cfg, {int(j): v for j, v in truth["coeffs"].items()},
                               np.asarray(meta["z_planes_um"]) + truth["true_z_offset_um"]).psf[k]
    fit_psf = build_psf_stack(cfg, fitted,
                              np.asarray(meta["z_planes_um"]) + exp["z_offset_um"]).psf[k]
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    for a, im, t in zip(ax, [true_psf, fit_psf, fit_psf - true_psf],
                        ["true PSF", "fitted PSF", "difference"]):
        if "difference" in t:
            v = max(1e-12, np.percentile(np.abs(im), 99))
            a.imshow(im, cmap="RdBu_r", vmin=-v, vmax=v)
        else:
            a.imshow(im / im.max(), cmap="inferno")
        a.set_title(t)
        a.axis("off")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def write_report(results_dir: str | Path) -> Path:
    results_dir = Path(results_dir)
    summary = json.loads((results_dir / "experiments.json").read_text())
    exps = summary["experiments"]
    report_dir = results_dir / "report"
    report_dir.mkdir(parents=True, exist_ok=True)

    _fig_coeffs(exps, report_dir / "fig_coeffs.png")
    _fig_snr(exps, report_dir / "fig_combined_snr.png")

    lines = [
        "# Synthetic Zernike Recovery — Report (Validation Tier 1)",
        "",
        f"Scene: {DEFAULT_SCENE.fov_px}² px @ {DEFAULT_SCENE.pixel_size_um} µm, "
        f"{DEFAULT_SCENE.n_planes} planes @ {DEFAULT_SCENE.z_step_um} µm, "
        f"fluorescent diameter {DEFAULT_SCENE.fluorescent_diameter_um*1000:.0f} nm. "
        f"Fitted modes: {', '.join(mode_name(j) for j in MODES)} "
        f"(Noll Z5–Z8; Z1/Z2/Z3/Z4 excluded per plan).",
        "",
        "| # | Experiment | Injected (RMS waves) | Fitted | RMS err | max |err| | z-offset (nm) | bg true→fit | Time |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for nm, rec in exps.items():
        inj = ", ".join(f"Z{k}={v:.2f}" for k, v in rec["truth_coeffs"].items()
                        if abs(float(v)) > 0) or "—"
        fit_s = ", ".join(f"Z{k}={v:+.3f}" for k, v in rec["fitted_coeffs"].items())
        lines.append(
            f"| {nm} | {rec.get('note','')} | {inj} | {fit_s} | "
            f"{rec['wavefront_rms_error']:.4f} | {rec['max_abs_err']:.4f} | "
            f"{rec['z_offset_um']*1000:+.0f} | "
            f"{rec['background_true']:.1f}→{rec['background_fitted']:.1f} | "
            f"{rec['fit_time_s']:.1f}s |")

    lines += ["", "## Pass criteria (plan §6)", ""]
    for nm, rec in exps.items():
        crits = rec["pass_criteria"]
        if not crits:
            continue
        status = []
        for c, d in crits.items():
            if d["pass"] is None:
                status.append(f"{c}: {d['value']} (quantified)")
            elif isinstance(d["value"], str):
                status.append(f"{c}: {d['value']} {'PASS' if d['pass'] else 'FAIL'}")
            else:
                status.append(f"{c}: {d['value']:.4f} {'PASS' if d['pass'] else 'FAIL'}")
        lines.append(f"- **{nm}**: " + "; ".join(status))
    lines += ["", f"**Overall: {'ALL PASS' if summary['all_pass'] else 'NOT ALL PASS'}**",
              "", "## Figures", "",
              "- ![](fig_coeffs.png) — fitted − injected per experiment",
              "- ![](fig_combined_snr.png) — experiment 3 RMS error vs SNR"]

    # per-experiment stack/PSF figures (spot check: exp 1, one single-mode, exp 3 snr50, exp 4)
    for nm in ("1_null", "2_single_Z6", "3_combined_snr50", "4_defocus_leakage"):
        if nm in exps:
            _fig_stack(nm, exps[nm], report_dir / f"fig_stacks_{nm}.png")
            _fig_psf(nm, exps[nm], report_dir / f"fig_psf_{nm}.png")
            lines.append(f"- ![](fig_stacks_{nm}.png) — {nm}: raw / fitted forward / residual "
                         f"(in-focus plane)")
            lines.append(f"- ![](fig_psf_{nm}.png) — {nm}: true vs fitted in-focus PSF")

    out = report_dir / "report.md"
    out.write_text("\n".join(lines) + "\n")
    print(f"[report] wrote {out}")
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Write report.md + figures from experiments.json")
    ap.add_argument("--results", default="results")
    args = ap.parse_args(argv)
    write_report(args.results)


if __name__ == "__main__":
    main()
