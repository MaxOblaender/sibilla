"""Semi-automatic rod extraction from 3D crops (spec §3 segments.py row).

Pipeline: Otsu threshold (MIP, fallback per-plane) -> 3D skeletonize
(`skimage.morphology.skeletonize(..., method="lee")` in scikit-image >= 0.22)
-> split at high curvature -> intensity-weighted PCA line fit per piece ->
(S, 6) segments [x0, y0, z0, x1, y1, z1] (px / plane units) + flags.

Flags (spec §0/§7 risks):
- "sheared": z-profile bimodal or fitted axis z-drift > 1 µm
  (live-cell motion suspicion — Review Focus 3)
- "saturated": any ROI pixel >= sat_level (16300)
- "edge": segment within 2 px of the crop border
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

_SHEAR_DRIFT_UM = 1.0
_EDGE_MARGIN_PX = 2.0


@dataclass
class SegmentsResult:
    segments: np.ndarray                       # (S, 6) [x0,y0,z0,x1,y1,z1]
    flags: dict[int, list[str]] = field(default_factory=dict)
    labels: np.ndarray | None = None           # (Nz, M, N) int labels


def _weighted_pca_axis(coords: np.ndarray,
                       weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Intensity-weighted PCA: returns (center, principal direction)."""
    w = weights / weights.sum()
    center = (coords * w[:, None]).sum(axis=0)
    d = coords - center
    cov = (d * w[:, None]).T @ d
    evals, evecs = np.linalg.eigh(cov)
    axis = evecs[:, -1]
    if axis[0] < 0:                     # canonical direction (+x first)
        axis = -axis
    return center, axis


def _fit_line_segment(coords: np.ndarray, weights: np.ndarray
                      ) -> np.ndarray:
    """Endpoints [x0, y0, z0, x1, y1, z1] of the weighted PCA segment.

    coords columns are (z, y, x) — plane index, row, column (build order).
    """
    center, axis = _weighted_pca_axis(coords, weights)
    t = (coords - center) @ axis
    tmin, tmax = t.min(), t.max()
    p0 = center + tmin * axis
    p1 = center + tmax * axis
    # coords are (z, y, x) -> emit [x, y, z] per the data contract
    return np.array([p0[2], p0[1], p0[0], p1[2], p1[1], p1[0]])


def _z_profile_bimodal(profile: np.ndarray) -> bool:
    """True if the z-profile has two peaks > 50% max separated by a
    valley < 50% of max (motion-shear / two-depth signature)."""
    p = profile / max(profile.max(), 1e-9)
    above = p >= 0.5
    if not above.any():
        return False
    # count runs of above
    runs = []
    start = None
    for i, a in enumerate(above):
        if a and start is None:
            start = i
        elif not a and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(above) - 1))
    return len(runs) >= 2


def extract_segments(stack: np.ndarray, pixel_um: float, z_step_um: float,
                     min_area_px: int = 30,
                     sat_level: float = 16300) -> SegmentsResult:
    """Extract rod segments from a (Nz, M, N) crop.

    Robust v1 (per the real-data plan's "semi-automatic"): one 3D connected
    component per rod — Otsu on the MIP gives the binary mask; components
    are labeled in 3D on (stack > mip_thr) within each MIP component's
    bbox; each component becomes ONE intensity-weighted PCA segment.
    Splitting touching rods (curvature) is deferred to manual CSV review.
    """
    from skimage import filters, measure

    stack = np.asarray(stack, dtype=np.float32)
    nz, m, n = stack.shape
    mip = stack.max(axis=0)
    thr = filters.threshold_otsu(mip)
    binary = mip > thr
    labels_mip = measure.label(binary)
    segments: list[np.ndarray] = []
    flags: dict[int, list[str]] = {}

    for region in measure.regionprops(labels_mip, intensity_image=mip):
        if region.area < min_area_px:
            continue
        minr, minc, maxr, maxc = region.bbox
        comp_mask = labels_mip[minr:maxr, minc:maxc] == region.label
        # split touching rods at skeleton corners (>60°), then one
        # intensity-weighted PCA segment per piece (plan Task 7)
        for mask2d in _split_by_curvature(comp_mask):
            seg, fl = _extract_one(stack, mask2d,
                                   (minr, minc, maxr, maxc), thr,
                                   z_step_um, sat_level)
            if seg is None:
                continue
            segments.append(seg)
            flags[len(segments) - 1] = fl

    segs = (np.asarray(segments, dtype=float) if segments
            else np.zeros((0, 6)))
    return SegmentsResult(segments=segs, flags=flags, labels=labels_mip)


def _fit_line_segment_yxz(coords: np.ndarray, weights: np.ndarray
                          ) -> np.ndarray:
    """Endpoints [x0, y0, z0, x1, y1, z1] from (V, 3) coords in (y, x, z)."""
    center, axis = _weighted_pca_axis(coords, weights)
    t = (coords - center) @ axis
    p0 = center + t.min() * axis
    p1 = center + t.max() * axis
    return np.array([p0[1], p0[0], p0[2], p1[1], p1[0], p1[2]])


def _extract_one(stack: np.ndarray, mask2d: np.ndarray, bbox: tuple,
                 thr: float, z_step_um: float,
                 sat_level: float) -> tuple[np.ndarray | None, list[str]]:
    """One sub-mask -> (segment, flags); None if too small."""
    nz, m, n = stack.shape
    minr, minc, maxr, maxc = bbox
    if not mask2d.any() or mask2d.sum() < 8:
        return None, []
    prof = np.array([stack[k][minr:maxr, minc:maxc][mask2d].max()
                     for k in range(nz)])
    zband = np.where(prof >= 0.5 * prof.max())[0]
    if len(zband) == 0:
        zband = np.array([int(np.argmax(prof))])
    vox, wts = [], []
    for k in zband:
        plane = stack[k][minr:maxr, minc:maxc]
        sel = mask2d & (plane > thr)
        if not sel.any():
            sel = mask2d
        ys, xs = np.nonzero(sel)
        vals = plane[ys, xs]
        vox.extend(zip(ys + minr, xs + minc, [k] * len(ys)))
        wts.extend(vals.tolist())
    coords = np.array(vox, dtype=float)          # (V, 3) (y, x, z)
    wts = np.array(wts, dtype=float)
    seg = _fit_line_segment_yxz(coords, wts)
    fl: list[str] = []
    if _z_profile_bimodal(prof):
        fl.append("sheared")
    if abs(seg[5] - seg[2]) * z_step_um > _SHEAR_DRIFT_UM:
        fl.append("sheared")
    if (stack[:, minr:maxr, minc:maxc][:, mask2d].max(axis=0)
            >= sat_level).any():
        fl.append("saturated")
    if (minr < _EDGE_MARGIN_PX or minc < _EDGE_MARGIN_PX or
            maxr > m - _EDGE_MARGIN_PX or maxc > n - _EDGE_MARGIN_PX):
        fl.append("edge")
    return seg, fl


def _split_by_curvature(mask2d: np.ndarray) -> list[np.ndarray]:
    """Split a 2D component's mask into pieces at skeleton corners (>60°).

    The skeleton pixels are ordered by projecting onto the component's PCA
    axis (a straight rod projects monotonically; greedy nearest-neighbor
    chains zigzag on diagonal skeletons and create false turns). Direction
    changes are measured between half-window means around each skeleton
    pixel. Returns sub-masks; single-piece components pass through.
    """
    from skimage import morphology

    skel = morphology.skeletonize(mask2d)
    if not skel.any():
        return [mask2d]
    ys, xs = np.nonzero(skel)
    pts = np.stack([xs, ys], axis=1).astype(float)   # (P, 2) (x, y)
    if len(pts) < 7:
        return [mask2d]
    center = pts.mean(axis=0)
    cov = (pts - center).T @ (pts - center)
    _, evecs = np.linalg.eigh(cov)
    axis = evecs[:, -1]
    t = (pts - center) @ axis
    order = np.argsort(t)
    chain = pts[order]

    # direction between half-windows of +-3 pixels around each chain point
    h = 3
    cuts = [0]
    for i in range(h, len(chain) - h):
        v1 = chain[i] - chain[i - h]
        v2 = chain[i + h] - chain[i]
        n1, n2 = np.hypot(*v1), np.hypot(*v2)
        if n1 == 0 or n2 == 0:
            continue
        cosang = float(np.clip((v1 @ v2) / (n1 * n2), -1.0, 1.0))
        if np.degrees(np.arccos(cosang)) > 60.0:
            cuts.append(i)
    cuts.append(len(chain))

    if len(cuts) == 2:
        return [mask2d]

    from scipy import ndimage

    # assign every skeleton pixel to its piece; then every component pixel
    # to the nearest piece's skeleton
    skel_pieces = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        piece = np.zeros_like(skel)
        p = chain[a:b].astype(int)
        piece[p[:, 1], p[:, 0]] = True
        if piece.any():
            skel_pieces.append(piece)
    dists = np.stack([ndimage.distance_transform_edt(~p) for p in skel_pieces])
    assign = np.argmin(dists, axis=0)
    out = []
    for j, piece in enumerate(skel_pieces):
        sub = mask2d & (assign == j)
        # a real rod piece is at least a rod-width across (>=30 px², matching
        # min_area_px); smaller fragments are noise, not rods
        if sub.sum() >= 30:
            out.append(sub)
    # if everything was fragmented away, keep the whole component (safer
    # than dropping a real rod because its skeleton is noisy)
    return out if out else [mask2d]


def write_segments_csv(path: str | Path, res: SegmentsResult) -> None:
    """Write segments + flags for manual review/correction."""
    path = Path(path)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["index", "x0", "y0", "z0", "x1", "y1", "z1", "flags"])
        for i, seg in enumerate(res.segments):
            w.writerow([i, *[f"{v:.3f}" for v in seg],
                        ";".join(res.flags.get(i, []))])


def read_segments_csv(path: str | Path) -> SegmentsResult:
    """Read back a (possibly hand-corrected) segments CSV."""
    path = Path(path)
    segs: list[np.ndarray] = []
    flags: dict[int, list[str]] = {}
    with open(path, newline="") as fh:
        r = csv.reader(fh)
        header = next(r)
        for row in r:
            if not row or row[0].strip().lower() == "index":
                continue
            i = int(row[0])
            vals = [float(v) for v in row[1:7]]
            segs.append(np.array(vals))
            fl = [f for f in row[7].split(";") if f] if len(row) > 7 else []
            flags[i] = fl
    return SegmentsResult(
        segments=np.asarray(segs, dtype=float) if segs else np.zeros((0, 6)),
        flags=flags, labels=None)


def make_review_figure(stack: np.ndarray, res: SegmentsResult,
                       out_png: str | Path) -> None:
    """MIP with numbered skeleton overlay; flagged segments in red."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    stack = np.asarray(stack, dtype=float)
    mip = stack.max(axis=0)
    fig, ax = plt.subplots(figsize=(7, 7))
    lo, hi = np.percentile(mip, [1, 99.5])
    ax.imshow(mip, cmap="gray", vmin=lo, vmax=hi)
    for i, seg in enumerate(res.segments):
        fl = res.flags.get(i, [])
        color = "red" if fl else "lime"
        ax.plot([seg[0], seg[3]], [seg[1], seg[4]],
                color=color, linewidth=1.5, alpha=0.9)
        ax.annotate(str(i), (seg[3], seg[4]), color=color, fontsize=9)
    ax.set_title(f"{len(res.segments)} segments · "
                 f"{sum(1 for f in res.flags.values() if f)} flagged")
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
