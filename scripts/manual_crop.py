"""Manual crop loop, step 2: given a region (top-left in um), cut a 3D window,
center a 15-plane band on the in-focus single rods, segment the band, and
produce a numbered review figure + metrics table for the user to pick from.

Usage (server):
  .venv/bin/python manual_crop.py --y <um> --x <um> [--w 256 --h 256] [--tag name]

Outputs (results/manual_crops/<tag>/):
  crop.tif            the 15-plane band (feed to fit_real_crop)
  review.png          MIP of the band, every segment numbered,
                      green = passes single-rod criteria, red = not,
                      label = "idx len/zspan"
  segments.csv        idx, endpoints, len_um, zspan_um, nn_um, zpeak, flags
  metrics.txt         the printed table

Single-rod criteria (calibrated on user crop -1):
  unflagged, 0.7 <= len <= 5.0 um, zspan <= 0.8 um, nn >= 0.6 um,
  z-peak >= 4 planes from the band edges.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import tifffile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, ".")
from sibilla.io import load_channel, square_pad
from sibilla.segments import extract_segments

CZI = "GoC-Exp4_C3_7.czi"
CH = 1
PX, ZSTEP = 0.1135, 0.5
BAND = 15
LEN_MIN, LEN_MAX = 0.7, 5.0
ZSPAN_MAX = 0.8
ISO_MIN = 0.6
ZEDGE = 4


def lap2(p):
    q = p.astype(np.float64)
    c = q[1:-1, 1:-1]
    return np.abs(q[2:, 1:-1] + q[:-2, 1:-1] + q[1:-1, 2:] +
                  q[1:-1, :-2] - 4.0 * c)


def focus_z(win):
    sharp = []
    for k in range(win.shape[0]):
        p = win[k].astype(np.float64)
        L = lap2(p)
        thr = np.percentile(p, 99)
        m = p[1:-1, 1:-1] >= thr
        sharp.append(L[m].mean() if m.any() else 0.0)
    s = np.convolve(np.asarray(sharp), np.ones(3) / 3, mode="same")
    return int(np.argmax(s))


def seg_metrics(res, win):
    segs = res.segments
    rows = []
    for i, s in enumerate(segs):
        L = np.hypot(s[3]-s[0], s[4]-s[1]) * PX
        zs = abs(s[5]-s[2]) * ZSTEP
        cx, cy = (s[0]+s[3])/2, (s[1]+s[4])/2
        nn = min((np.hypot((o[0]+o[3])/2-cx, (o[1]+o[4])/2-cy)
                 for j, o in enumerate(segs) if j != i), default=99.0) * PX
        y0 = max(0, int(min(s[1], s[4])))
        y1 = min(win.shape[1], int(max(s[1], s[4])) + 1)
        x0 = max(0, int(min(s[0], s[3])))
        x1 = min(win.shape[2], int(max(s[0], s[3])) + 1)
        prof = win[:, y0:y1, x0:x1].max(axis=(1, 2))
        zpeak = int(prof.argmax())
        fl = res.flags.get(i, [])
        rows.append(dict(i=i, seg=s, L=L, zs=zs, nn=nn, zpeak=zpeak,
                         fl=fl))
    return rows


def is_single(r, z0):
    ok = (not r["fl"] and LEN_MIN <= r["L"] <= LEN_MAX
          and r["zs"] <= ZSPAN_MAX and r["nn"] >= ISO_MIN)
    margin = (r["zpeak"] - z0 >= ZEDGE
              and (z0 + BAND - 1) - r["zpeak"] >= ZEDGE)
    return ok and margin, ok, margin


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--y", type=float, required=True, help="top-left y in um")
    ap.add_argument("--x", type=float, required=True, help="top-left x in um")
    ap.add_argument("--w", type=int, default=256)
    ap.add_argument("--h", type=int, default=256)
    ap.add_argument("--tag", default="crop")
    ap.add_argument("--z", type=int, default=None,
                    help="absolute in-focus z plane (overrides auto-center)")
    args = ap.parse_args()

    outdir = f"results/manual_crops/{args.tag}"
    os.makedirs(outdir, exist_ok=True)
    stack = load_channel(CZI, CH)
    nz, H, W = stack.shape

    y0 = int(round(args.y / PX))
    x0 = int(round(args.x / PX))
    y0 = max(0, min(y0, H - args.h))
    x0 = max(0, min(x0, W - args.w))
    win = stack[:, y0:y0+args.h, x0:x0+args.w]
    print(f"region: y={args.y}um x={args.x}um -> px ({y0},{x0}) "
          f"window {win.shape}", flush=True)

    # full-window metrics -> choose band center from in-focus singles
    # (NO zspan criterion here: in the full 196-plane window the PSF halo
    # inflates z-span; zspan only makes sense inside a thin band)
    full_res = extract_segments(square_pad(win), PX, ZSTEP)
    full_rows = seg_metrics(full_res, win)
    prelim = [r for r in full_rows
              if not r["fl"] and LEN_MIN <= r["L"] <= LEN_MAX
              and r["nn"] >= ISO_MIN]
    if args.z is not None:
        zc = int(args.z)
        why = "user-given focus z"
    elif prelim:
        zc = int(np.median([r["zpeak"] for r in prelim]))
        why = f"median z-peak of {len(prelim)} in-focus singles"
    else:
        # brightness-profile peak (rods are big bright areas; dust is tiny)
        prof = (win > 4 * win.max(axis=(1, 2)).mean()).sum(axis=(1, 2))
        zc = int(np.argmax(prof))
        why = "brightness-profile peak (no singles pre-selected)"
    z0 = max(0, min(zc - BAND//2, nz - BAND))
    band = win[z0:z0+BAND]
    print(f"band: z {z0}..{z0+BAND-1}  ({why})", flush=True)
    tifffile.imwrite(f"{outdir}/crop.tif", band)

    # segment the BAND (this is what the fitter will see; indices match)
    res = extract_segments(square_pad(band), PX, ZSTEP)
    rows = seg_metrics(res, band)
    tbl = []
    for r in rows:
        ok, base_ok, margin = is_single(r, z0)
        why = []
        if r["fl"]:
            why.append(";".join(r["fl"]))
        if not (LEN_MIN <= r["L"] <= LEN_MAX):
            why.append(f"len{r['L']:.1f}")
        if r["zs"] > ZSPAN_MAX:
            why.append(f"zspan{r['zs']:.1f}")
        if r["nn"] < ISO_MIN:
            why.append(f"nn{r['nn']:.1f}")
        if base_ok and not margin:
            why.append("band-edge")
        r["ok"], r["why"] = ok, " ".join(why)
        tbl.append(r)
    tbl.sort(key=lambda r: not r["ok"])

    # review figure
    padded = square_pad(band)
    mip = padded.max(axis=0)
    # square_pad is TOP-LEFT aligned (out[:, :m, :n] = stack), so segment
    # coords are already in display coords — no offset (the old centered
    # offset shifted lines down for non-square crops).
    poff, poxx = 0, 0
    fig, ax = plt.subplots(figsize=(10, 10))
    lo, hi = np.percentile(mip, [1, 99.5])
    ax.imshow(mip, cmap="gray", vmin=lo, vmax=hi)
    for r in tbl:
        s = r["seg"]
        col = "lime" if r["ok"] else "red"
        ax.plot([s[0]+poxx, s[3]+poxx], [s[1]+poff, s[4]+poff],
                color=col, lw=2.0, alpha=0.95)
        ax.annotate(f"{r['i']} {r['L']:.1f}/{r['zs']:.1f}",
                    (s[3]+poxx, s[4]+poff), color=col, fontsize=9,
                    weight="bold")
    n_ok = sum(1 for r in tbl if r["ok"])
    ax.set_title(f"{args.tag}: y={args.y}um x={args.x}um  band z{z0}..{z0+BAND-1}"
                 f" ({z0*0.5:.1f}..{z0*0.5+(BAND-1)*0.5:.1f}um rel)\n"
                 f"{n_ok} single rods (green) of {len(tbl)} segments — "
                 f"pick by index", fontsize=11)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(f"{outdir}/review.png", dpi=110)

    # csv + table
    with open(f"{outdir}/segments.csv", "w") as fh:
        fh.write("idx,x0,y0,z0,x1,y1,z1,len_um,zspan_um,nn_um,zpeak,flags,ok\n")
        for r in rows:
            s = r["seg"]
            fh.write(f"{r['i']},{s[0]:.2f},{s[1]:.2f},{s[2]:.1f},"
                     f"{s[3]:.2f},{s[4]:.2f},{s[5]:.1f},"
                     f"{r['L']:.2f},{r['zs']:.2f},{r['nn']:.2f},"
                     f"{r['zpeak']},{'|'.join(r['fl'])},{int(r['ok'])}\n")
    lines = [f"{args.tag}: region (y={args.y}um, x={args.x}um), "
             f"band z{z0}..{z0+BAND-1}"]
    lines.append(f"{'idx':>4} {'len':>5} {'zspan':>6} {'nn':>5} {'zpk':>4}  "
                 f"{'SINGLE?':8} why")
    for r in tbl:
        lines.append(f"{r['i']:4d} {r['L']:5.2f} {r['zs']:6.2f} {r['nn']:5.2f} "
                     f"{r['zpeak']:4d}  {'YES' if r['ok'] else 'no':8} "
                     f"{r['why']}")
    txt = "\n".join(lines)
    open(f"{outdir}/metrics.txt", "w").write(txt + "\n")
    print(txt, flush=True)
    json.dump(dict(tag=args.tag, y_um=args.y, x_um=args.x, y0=y0, x0=x0,
                   z0=z0, z1=z0+BAND-1, n_segments=len(rows),
                   n_single=sum(1 for r in tbl if r["ok"])),
              open(f"{outdir}/crop_info.json", "w"), indent=1)
    print(f"DONE -> {outdir}/ (crop.tif, review.png, segments.csv, metrics.txt)",
          flush=True)


if __name__ == "__main__":
    main()
