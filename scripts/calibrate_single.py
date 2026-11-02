"""Calibrate single-bacterium criteria on the USER's own crop (-1), which
already produced the accepted 4-rod fit. Measure length / z-span / isolation /
width of every segment, and of the clean ones, so the numeric thresholds are
set from the user's trusted data.
"""
import sys
import numpy as np
import tifffile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

sys.path.insert(0, ".")
from sibilla.io import square_pad
from sibilla.segments import extract_segments
from sibilla.metrics import apparent_diameter_um

PX, ZSTEP = 0.1135, 0.5

def analyze(stack, title):
    padded = square_pad(stack)
    res = extract_segments(padded, PX, ZSTEP)
    segs = res.segments
    mip = padded.max(axis=0)
    poff = (padded.shape[1] - stack.shape[1]) // 2
    poxx = (padded.shape[2] - stack.shape[2]) // 2
    print(f"\n=== {title}  padded {padded.shape}  segments {len(segs)}")
    rows = []
    for i, s in enumerate(segs):
        L = np.hypot(s[3]-s[0], s[4]-s[1]) * PX
        ZSPAN = abs(s[5]-s[2]) * ZSTEP
        cx, cy = (s[0]+s[3])/2 + poxx, (s[1]+s[4])/2 + poff
        d = np.hypot(s[3]-s[0], s[4]-s[1])
        nn = min((np.hypot((o[0]+o[3])/2-cx, (o[1]+o[4])/2-cy)
                 for j, o in enumerate(segs) if j != i), default=99) * PX
        w = apparent_diameter_um(mip, PX, center=(cx, cy),
                                 direction=((s[3]-s[0])/d, (s[4]-s[1])/d) if d else (0,1))
        fl = res.flags.get(i, [])
        clean = not fl
        rows.append(dict(i=i, L=L, W=w, zspan=ZSPAN, nn=nn,
                         cx=cx, cy=cy, clean=clean, fl=fl))
        print(f"  seg {i:2d} {'CLEAN' if clean else 'FLAG ':5} "
              f"len={L:4.2f}um  w={w if np.isfinite(w) else float('nan'):4.2f}um  "
              f"zspan={ZSPAN:4.2f}um  nn={nn:4.2f}um  {(';'.join(fl)) if fl else ''}")
    cl = [r for r in rows if r["clean"]]
    print(f"\n  CLEAN segments ({len(cl)}):")
    if cl:
        Ls = np.array([r['L'] for r in cl]); Ws = np.array([r['W'] for r in cl if np.isfinite(r['W'])])
        zs = np.array([r['zspan'] for r in cl]); nns = np.array([r['nn'] for r in cl])
        print(f"    length   min={Ls.min():.2f} med={np.median(Ls):.2f} max={Ls.max():.2f} um")
        if len(Ws): print(f"    width    min={Ws.min():.2f} med={np.median(Ws):.2f} max={Ws.max():.2f} um")
        print(f"    z-span   min={zs.min():.2f} med={np.median(zs):.2f} max={zs.max():.2f} um")
        print(f"    isolation min={nns.min():.2f} med={np.median(nns):.2f} max={nns.max():.2f} um")
    # figure
    fig, ax = plt.subplots(figsize=(10, 9))
    lo, hi = np.percentile(mip, [1, 99.5])
    ax.imshow(mip, cmap="gray", vmin=lo, vmax=hi)
    for r in rows:
        s = segs[r["i"]]
        col = "lime" if r["clean"] else "red"
        ax.plot([s[0]+poxx, s[3]+poxx], [s[1]+poff, s[4]+poff], color=col, lw=2.2, alpha=0.9)
        ax.annotate(f"{r['i']} {r['L']:.1f}x{r['zspan']:.1f}",
                    (s[3]+poxx, s[4]+poff), color=col, fontsize=8)
    ax.set_title(f"{title}\ngreen=clean  label: segId len(um)x z-span(um)")
    ax.axis("off")
    fig.tight_layout()
    tag = title.split(" ")[0].replace("GoC-Exp4_C3_7-", "")
    fig.savefig(f"results_local/calib_{tag}.png", dpi=110)
    print(f"  wrote results_local/calib_{tag}.png")
    return rows

import os
os.makedirs("results_local", exist_ok=True)
# the user's good crop
analyze(tifffile.imread("GoC-Exp4_C3_7-1_3d.tif").astype(np.float32),
        "user crop -1 (4-rod fit)")
