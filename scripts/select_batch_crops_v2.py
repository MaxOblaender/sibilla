"""Select a SMALL batch of crops with WELL-ISOLATED SINGLE rods, using criteria
calibrated on the user's own good crop (-1), which gave the accepted 4-rod fit.

Per-segment "good single rod" criteria (from user crop -1):
  - unflagged (no sheared/edge/saturated)
  - flat:        z-span (|dz of the fitted line|)  <= ZSPAN_MAX   (~0.8 um)
  - single cell: LEN_MIN <= length <= LEN_MAX        (~0.7-5.0 um)
  - isolated:    nearest-neighbour centre dist >= ISO_MIN        (~0.6 um)
  - not clipped: the rod's own z-peak is >= ZEDGE planes inside the 15-band

Window acceptance:
  - 3..MAXRODS good single rods
  - good rods' z-peaks all within ZSPREAD planes (one band can hold them)
  - penalise total density (many segments = "too much going on")
  - pick top SCORE, spatially spread, non-overlapping, outside old fits.

Outputs: results/fit_batch2/contact_sheet.png, picks.json  (server).
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
from sibilla.io import load_channel, square_pad
from sibilla.segments import extract_segments

CZI = "GoC-Exp4_C3_7.czi"
CH = 1
PX, ZSTEP = 0.1135, 0.5
WIN = 256
BAND = 15
OUT = "results/fit_batch2"

# calibrated thresholds (user crop -1)
LEN_MIN, LEN_MAX = 0.7, 5.0
ZSPAN_MAX = 0.8          # um, flat single depth
ISO_MIN = 0.6            # um, isolation to nearest rod centre
ZEDGE = 4                # planes, rod z-peak must be this far from band edge
ZSPREAD = 7              # planes, good rods must fit within this depth range
N_GOOD_MIN, N_GOOD_MAX = 3, 6

TAKEN = [(1100, 116, 256, 184), (640, 640, 256, 256),
         (768, 576, 256, 256), (832, 576, 256, 256)]


def overlap(a, b):
    y0, x0, h, w = a
    y1, x1, h2, w2 = b
    return not (y0 + h <= y1 or y1 + h2 <= y0 or x0 + w <= x1 or x1 + w2 <= x0)


def rod_zpeak(win, seg):
    y0, x0 = int(min(seg[1], seg[4])), int(min(seg[0], seg[3]))
    y1, x1 = int(max(seg[1], seg[4])) + 1, int(max(seg[0], seg[3])) + 1
    y0, y1 = max(0, y0), min(win.shape[1], y1)
    x0, x1 = max(0, x0), min(win.shape[2], x1)
    prof = win[:, y0:y1, x0:x1].max(axis=(1, 2))
    return int(prof.argmax())


def score_window(win):
    padded = square_pad(win)
    try:
        res = extract_segments(padded, PX, ZSTEP)
    except Exception:
        return None
    segs = res.segments
    n_tot = len(segs)
    if n_tot == 0:
        return None
    good = []
    for i, s in enumerate(segs):
        if res.flags.get(i):
            continue
        L = np.hypot(s[3]-s[0], s[4]-s[1]) * PX
        zs = abs(s[5]-s[2]) * ZSTEP
        if not (LEN_MIN <= L <= LEN_MAX):
            continue
        if zs > ZSPAN_MAX:
            continue
        cx, cy = (s[0]+s[3])/2, (s[1]+s[4])/2
        nn = min((np.hypot((o[0]+o[3])/2-cx, (o[1]+o[4])/2-cy)
                 for j, o in enumerate(segs) if j != i), default=99) * PX
        if nn < ISO_MIN:
            continue
        zp = rod_zpeak(win, s)
        good.append(dict(i=i, L=L, zs=zs, nn=nn, zpeak=zp))
    return dict(n_tot=n_tot, good=good)


def main():
    stack = load_channel(CZI, CH)
    nz, H, W = stack.shape
    print("channel", stack.shape, flush=True)
    # FULL grid scan (seed file used the OLD loose criteria and would
    # exclude windows with >8 old-clean but many good singles)
    results = []
    nwin = 0
    for y0 in range(0, H - WIN + 1, 64):
        for x0 in range(0, W - WIN + 1, 64):
            win = stack[:, y0:y0+WIN, x0:x0+WIN]
            r = score_window(win)
            nwin += 1
            if r:
                r["y"], r["x"] = y0, x0
                results.append(r)
    print(f"scored {nwin} windows, {len(results)} with segments", flush=True)

    def window_score(r):
        g = r["good"]
        if not (N_GOOD_MIN <= len(g) <= N_GOOD_MAX):
            return None
        zps = [x["zpeak"] for x in g]
        if max(zps) - min(zps) > ZSPREAD:
            return None
        # band centred on median z-peak
        zc = int(np.median(zps))
        z0 = max(0, min(zc - BAND//2, nz - BAND))
        inband = [x for x in g if x["zpeak"] - z0 >= ZEDGE
                  and z0 + BAND - 1 - x["zpeak"] >= ZEDGE]
        if len(inband) < N_GOOD_MIN:
            return None
        iso = np.median([x["nn"] for x in inband])
        flat = np.median([x["zs"] for x in inband])
        density = r["n_tot"]
        return (len(inband), iso, -flat, -density, len(inband)/density)

    scored = []
    for r in results:
        s = window_score(r)
        if s is not None:
            scored.append((s, r))
    scored.sort(key=lambda t: t[0], reverse=True)
    print(f"{len(scored)} windows pass criteria", flush=True)

    picks = []
    for s, r in scored:
        win = (r["y"], r["x"], WIN, WIN)
        if any(overlap(win, t) for t in TAKEN):
            continue
        if any(overlap(win, (p["y"], p["x"], WIN, WIN)) for p in picks):
            continue
        picks.append((s, r))
        if len(picks) == 5:
            break

    os.makedirs(OUT, exist_ok=True)
    info = []
    for k, (s, r) in enumerate(picks, start=1):
        y0, x0 = r["y"], r["x"]
        win = stack[:, y0:y0+WIN, x0:x0+WIN]
        g = r["good"]
        zc = int(np.median([x["zpeak"] for x in g]))
        z0 = max(0, min(zc - BAND//2, nz - BAND))
        band = win[z0:z0+BAND]
        name = f"crop_c{k:02d}"
        tifffile.imwrite(f"{OUT}/{name}.tif", band)
        info.append(dict(name=name, y=y0, x=x0, z0=z0, z1=z0+BAND-1,
                         focus_z=zc, n_good=len(g), n_tot=r["n_tot"],
                         score=[round(float(v),3) for v in s],
                         rods=[dict(L=round(x["L"],2), zspan=round(x["zs"],2),
                                     nn=round(x["nn"],2), zpeak=x["zpeak"])
                                for x in g]))
        print(f"{name}: y={y0} x={x0} focus_z={zc} good={len(g)} tot={r['n_tot']} "
              f"band z{z0}..{z0+BAND-1}", flush=True)
    json.dump(info, open(f"{OUT}/picks.json", "w"), indent=1)

    # ---- contact sheet: MIP of band + good-rod overlay + metrics ----
    fig, axes = plt.subplots(1, max(1, len(picks)),
                             figsize=(5.4*max(1, len(picks)), 5.6))
    axes = np.atleast_1d(axes)
    for ax, it in zip(axes, info):
        band = tifffile.imread(f"{OUT}/{it['name']}.tif")
        mip = band.max(axis=0)
        lo, hi = np.percentile(mip, [1, 99.5])
        ax.imshow(mip, cmap="gray", vmin=lo, vmax=hi)
        res = extract_segments(square_pad(band), PX, ZSTEP)
        # recompute good set on the band for overlay (approx: same rods)
        for rd in it["rods"]:
            pass
        # overlay by re-scoring the band
        rb = score_window(band)
        if rb:
            for g in rb["good"]:
                s = res.segments[g["i"]]
                ax.plot([s[0], s[3]], [s[1], s[4]], color="lime", lw=2.0)
        ax.set_title(f"{it['name']}  y={it['y']} x={it['x']}\n"
                     f"focus z={it['focus_z']} ({it['focus_z']*0.5:.1f}um)\n"
                     f"{it['n_good']} isolated singles / {it['n_tot']} segs\n"
                     f"len " + (", ".join(f"{x['L']:.1f}" for x in it["rods"][:6]))
                     + f"\niso " + (", ".join(f"{x['nn']:.1f}" for x in it["rods"][:6])),
                     fontsize=8)
        ax.axis("off")
    for ax in axes[len(picks):]:
        ax.axis("off")
    fig.suptitle("Fit batch v2 — isolated SINGLE rods only (calibrated on your crop -1). "
                 "green = good single rod. Pick 2-3 by eye.", fontsize=12)
    fig.tight_layout()
    fig.savefig(f"{OUT}/contact_sheet.png", dpi=110)
    print(f"WROTE {OUT}/contact_sheet.png", flush=True)
    print("BATCH2 DONE", flush=True)


main()
