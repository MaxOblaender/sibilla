"""Sanity test for a set of user-made crop TIFFs (4,5,6 + 1,2,3).

For each crop:
  - NCC check: is it an exact sub-block extract of the CZI channel? (catches
    the orientation/transpose trap before any fit)
  - segment the band, recompute the calibrated single-rod metrics
  - write a labeled review figure (green=passes single-rod criteria) + table

Outputs: results/crop_sanity/<name>/{review.png,metrics.txt} and
         results/crop_sanity/contact_sheet.png
"""
import json
import os
import sys

import numpy as np
import tifffile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, ".")
from sibilla.io import load_channel, read_crop_tiff, square_pad, locate_stack
from sibilla.segments import extract_segments

PX, ZSTEP = 0.1135, 0.5
LEN_MIN, LEN_MAX = 0.7, 5.0
ZSPAN_MAX = 0.8
ISO_MIN = 0.6
ZEDGE = 4

CROPS = [f"GoC-Exp4_C3_7-{k}_3d.tif" for k in (1, 2, 3, 4, 5, 6)]
OUT = "results/crop_sanity"


def seg_table(res, band):
    segs = res.segments
    nz = band.shape[0]
    rows = []
    for i, s in enumerate(segs):
        L = np.hypot(s[3]-s[0], s[4]-s[1]) * PX
        zs = abs(s[5]-s[2]) * ZSTEP
        cx, cy = (s[0]+s[3])/2, (s[1]+s[4])/2
        nn = min((np.hypot((o[0]+o[3])/2-cx, (o[1]+o[4])/2-cy)
                 for j, o in enumerate(segs) if j != i), default=99.0) * PX
        y0 = max(0, int(min(s[1], s[4]))); y1 = min(band.shape[1], int(max(s[1], s[4]))+1)
        x0 = max(0, int(min(s[0], s[3]))); x1 = min(band.shape[2], int(max(s[0], s[3]))+1)
        zpeak = int(band[:, y0:y1, x0:x1].max(axis=(1, 2)).argmax())
        fl = res.flags.get(i, [])
        base = (not fl and LEN_MIN <= L <= LEN_MAX and zs <= ZSPAN_MAX
                and nn >= ISO_MIN)
        margin = zpeak >= ZEDGE and (nz - 1 - zpeak) >= ZEDGE
        why = []
        if fl: why.append("+".join(fl))
        if not (LEN_MIN <= L <= LEN_MAX): why.append(f"len{L:.1f}")
        if zs > ZSPAN_MAX: why.append(f"zs{zs:.1f}")
        if nn < ISO_MIN: why.append(f"nn{nn:.1f}")
        if base and not margin: why.append("zbnd")
        rows.append(dict(i=i, s=s, L=L, zs=zs, nn=nn, zpeak=zpeak, fl=fl,
                         ok=base and margin, why=" ".join(why)))
    return rows


def main():
    os.makedirs(OUT, exist_ok=True)
    stack = load_channel("GoC-Exp4_C3_7.czi", 1)
    summary, panels = [], []
    for name in CROPS:
        if not os.path.exists(name):
            print(f"MISSING {name}")
            continue
        crop, tags = read_crop_tiff(name)
        nz, m, n = crop.shape
        # NCC: locate this crop's middle plane in the channel (orientation check)
        mid = nz // 2
        ncc, y0, x0 = locate_stack(stack, crop[mid])
        band = square_pad(crop)
        res = extract_segments(band, PX, ZSTEP)
        rows = seg_table(res, band)
        rows.sort(key=lambda r: not r["ok"])
        d = f"{OUT}/{name.replace('.tif','')}"
        os.makedirs(d, exist_ok=True)
        n_ok = sum(1 for r in rows if r["ok"])
        lines = [f"{name}: shape {crop.shape}  NCC(direct)={ncc:.3f} at "
                 f"(y={y0}, x={x0})", f"{n_ok} single rods / {len(rows)} segments",
                 f"{'idx':>4} {'len':>5} {'zspan':>6} {'nn':>5} {'zpk':>4}  SINGLE?  why"]
        for r in rows:
            lines.append(f"{r['i']:4d} {r['L']:5.2f} {r['zs']:6.2f} {r['nn']:5.2f} "
                         f"{r['zpeak']:4d}  {'YES' if r['ok'] else 'no':8} {r['why']}")
        open(f"{d}/metrics.txt", "w").write("\n".join(lines) + "\n")
        # figure (square_pad is TOP-LEFT aligned: out[:, :m, :n] = stack,
        # so segments from extract_segments are ALREADY in display coords —
        # no offset. The old centered offset shifted lines down for
        # non-square crops, e.g. 36px for the 184-row crop -1.)
        mip = band.max(axis=0)
        poff, poxx = 0, 0
        fig, ax = plt.subplots(figsize=(9, 9 * m / max(n, 1)))
        lo, hi = np.percentile(mip, [1, 99.5])
        ax.imshow(mip, cmap="gray", vmin=lo, vmax=hi)
        for r in rows:
            s = r["s"]
            col = "lime" if r["ok"] else "red"
            ax.plot([s[0]+poxx, s[3]+poxx], [s[1]+poff, s[4]+poff],
                    color=col, lw=2.0)
            ax.annotate(f"{r['i']} {r['L']:.1f}/{r['zs']:.1f}",
                        (s[3]+poxx, s[4]+poff), color=col, fontsize=9,
                        weight="bold")
        ax.set_title(f"{name}  NCC={ncc:.2f}  {n_ok} single / {len(rows)} seg", fontsize=11)
        ax.axis("off")
        fig.tight_layout(); fig.savefig(f"{d}/review.png", dpi=100); plt.close(fig)
        summary.append(dict(name=name, shape=list(crop.shape), ncc=float(ncc),
                            y0=int(y0), x0=int(x0), n_seg=len(rows), n_single=n_ok,
                            singles=[dict(i=r["i"], L=round(r["L"],2),
                                          zs=round(r["zs"],2), nn=round(r["nn"],2))
                                     for r in rows if r["ok"]]))
        panels.append((name, band, rows, ncc))
        print(f"\n{name}: {n_ok} single / {len(rows)} seg  NCC={ncc:.3f}")
        for r in rows:
            print(f"   {r['i']:3d} len={r['L']:4.2f} zs={r['zs']:4.2f} "
                  f"nn={r['nn']:5.2f} zpk={r['zpeak']:3d} "
                  f"{'YES' if r['ok'] else 'no ':4} {r['why']}")

    json.dump(summary, open(f"{OUT}/summary.json", "w"), indent=1)

    # combined sheet
    ncol = 3
    nrow = (len(panels) + ncol - 1) // ncol
    fig, axs = plt.subplots(nrow, ncol, figsize=(6*ncol, 6*nrow))
    axs = np.atleast_1d(axs).ravel()
    for ax, (name, band, rows, ncc) in zip(axs, panels):
        mip = band.max(axis=0)
        lo, hi = np.percentile(mip, [1, 99.5])
        ax.imshow(mip, cmap="gray", vmin=lo, vmax=hi)
        for r in rows:
            s = r["s"]
            col = "lime" if r["ok"] else "red"
            ax.plot([s[0], s[3]], [s[1], s[4]], color=col, lw=1.8)
        n_ok = sum(1 for r in rows if r["ok"])
        ax.set_title(f"{name}\nNCC={ncc:.2f}  {n_ok} single / {len(rows)} seg", fontsize=9)
        ax.axis("off")
    for ax in axs[len(panels):]:
        ax.axis("off")
    fig.suptitle("User crops 1-6 — green = single rod (calibrated), red = rejected", fontsize=13)
    fig.tight_layout(); fig.savefig(f"{OUT}/contact_sheet.png", dpi=100)
    print(f"\nWROTE {OUT}/contact_sheet.png and summary.json")
    print("SANITY DONE")


main()