"""Conidia spot detector with mandatory sanity-check visuals (spec §8, §8.1).

Phase 2 module — **not** used by the milestone-4 pilot. Its job is the
empirical PSF sanity check of §8: find isolated, PSF-sized bright spots in a
L1-normalized CH1 tile and hand them to the maintainer for eyeball review.
History is the reason §8.1 is strict: auto-detections only enter a fit after
the sanity visuals below have been inspected.

Detection (:func:`detect`):

1. **Background / noise** from :func:`sibilla.twod.flatfield.estimate_empty_background`
   (robust MAD over the lowest-50 % subset, so spots cannot inflate sigma).
2. **Seeds** — local maxima of ``img`` (3×3 neighbourhood) strictly above
   ``bg + thresholds · sigma_bg``. Seeds are labelled with 8-connectivity so a
   plateau (several equal-valued pixels) yields ONE candidate rather than a
   cluster of duplicates. All seeds are kept, even ones later rejected, so the
   candidate list records what was looked at.
3. **Geometry** — per seed, the connected region (8-connectivity) above
   ``bg + (peak - bg) / 2`` *containing the seed* gives

   - ``radius_px``: equivalent radius ``sqrt(n / pi)`` of that region's area,
   - ``aspect``: axis ratio ``sqrt(lambda_max / lambda_min)`` of the
     intensity-weighted (``w = clip(img - bg, 0)``) second-moment tensor,
     so the major axis is always the numerator — a horizontal hypha and a
     vertical one both report aspect > 1, never < 1,
   - ``(x, y)``: the intensity-weighted centroid of that region.

   The half-max region does double duty: a seed on a smooth feature's slope
   is confined to the region around ITS peak (the low side falls below
   half-max), so a bright colony cannot "capture" a neighbour's region, and
   because ``(x, y)`` is reported at the centroid, two maxima on one flat
   plateau resolve to the same point and collapse to a single candidate.

4. **Gating**, in this priority (one reason per rejected candidate, most
   structural first): ``too_large`` (``radius_px > max_radius_px`` — anything
   bigger than a conidium is colony/hypha mass), then ``elongated``
   (``aspect > max_aspect`` — hyphae fragments), then ``merged`` (the
   candidate's half-max region comes within ``merge_px`` of another
   candidate's region — a blend, not two isolated objects), then ``too_dim``
   (peak not above the required excess — a re-check that cannot fail for
   :func:`detect`'s own seeds, but guards spot lists supplied any other way).
   Colonies and hyphae are therefore rejected *by construction*: a
   PSF-calibration object must be smaller than the PSF, and the size gate is
   the only one that can admit an extended-but-isotropic object.

   ``score = (peak - bg) / sigma_bg`` (peak SNR). Candidates are returned
   ordered by score, strongest first — the same confidence order the §8.1
   contact sheet uses, so a colony outscoring the real conidia is the first
   thing the maintainer sees.

:func:`write_sanity_visuals` produces the three §8.1 artifacts:
tile overlay, contact sheet and run summary (PNG + CSV). Rejected candidates
are drawn too — false negatives are as informative as false positives.
:func:`load_manual` reads the maintainer's manual override, which takes
priority over any detection.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, fields
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: no display in batch runs; must precede pyplot

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle, Rectangle  # noqa: E402
from scipy import ndimage  # noqa: E402

from sibilla.twod.flatfield import estimate_empty_background  # noqa: E402

__all__ = [
    "Spot",
    "DetectionParams",
    "detect",
    "load_manual",
    "write_sanity_visuals",
    "REASONS",
    "REASON_COLORS",
]

_EIGHT = np.ones((3, 3), dtype=int)  # 8-connectivity
_TILE_PNG = "spots_tile_{tile_id}.png"
_SHEET_PNG = "spots_sheet.png"
_SUMMARY_PNG = "spots_summary.png"
_SUMMARY_CSV = "spots_summary.csv"

#: Rejection reasons, most structural first (also the gating priority).
REASONS = ("too_large", "elongated", "merged", "too_dim")

#: Colour per rejection reason for the tile overlay (orange family, per §8.1),
#: plus green for accepted spots. "other" catches anything unexpected (e.g.
#: a spot list from a manual override) so nothing is ever drawn in silence.
REASON_COLORS = {
    "too_large": "#e8590c",  # dark orange — colony / cell mass
    "elongated": "#f59f00",  # amber — hypha fragment
    "merged": "#d6336c",  # magenta — overlaps a neighbour
    "too_dim": "#adb5bd",  # grey — noise-band peak
    "other": "#c0a000",  # olive — unclassified rejection
}

_SHEET_THUMB = 48  # px, extracted thumbnail side (spec §8.1)
_SHEET_ZOOM = 4  # thumbnails are upscaled ×4 on the contact sheet
_SHEET_CAPTION = 20  # px caption strip under each thumbnail
_SHEET_COLS = 8
_CSV_COLS = (
    "tile_id",
    "spot_id",
    "x",
    "y",
    "score",
    "radius_px",
    "aspect",
    "accepted",
    "reason",
)


@dataclass
class Spot:
    """One candidate spot; ``accepted=False`` carries a ``reason``.

    ``x``, ``y`` are pixel coordinates (``x`` = column, ``y`` = row, so
    ``img[int(round(y)), int(round(x))]`` is the spot). ``score`` is the peak
    SNR ``(peak - bg) / sigma_bg``; ``radius_px`` is the equivalent radius and
    ``aspect`` the second-moment axis ratio of the spot's half-max region.
    """

    x: float
    y: float
    score: float
    radius_px: float
    aspect: float
    accepted: bool
    reason: str = ""


@dataclass
class DetectionParams:
    """Detector parameters (spec §8).

    ``thresholds`` is the seed threshold ``k`` in ``bg + k·sigma_bg``;
    ``max_radius_px`` and ``max_aspect`` are the size and isotropy gates
    (colonies and hyphae are rejected by construction); ``merge_px`` is the
    minimum distance that must separate two candidates' half-max regions —
    closer than that they are one blend (``merged``).
    """

    thresholds: float = 6.0
    max_radius_px: float = 8.0
    max_aspect: float = 1.8
    merge_px: float = 3


def _half_max_region(
    img: np.ndarray, bg: float, y: int, x: int, window: int | None = None
) -> tuple[np.ndarray, tuple[int, int], tuple[int, int, int, int]]:
    """Connected region above half-max containing (y, x), cropped to its bbox.

    Returns ``(region, (oy, ox), (by0, by1, bx0, bx1))`` where ``region`` is
    the boolean mask cropped to its bounding box, ``(oy, ox)`` is the crop's
    origin in image coordinates and the bbox is inclusive. With ``window``,
    labeling runs on a ``(2·window + 1)²`` crop around the seed — any region
    small enough to be a legal spot fits far inside it, so the common case
    avoids labelling the whole tile. A region that REACHES the crop border
    may continue beyond it (big colony), so the full-image labelling is
    redone for those; correctness never depends on the window.
    """
    peak = float(img[y, x])
    level = bg + 0.5 * (peak - bg)

    def _pack(sub: np.ndarray, oy: int, ox: int):
        ys, xs = np.nonzero(sub)
        by0, by1 = oy + int(ys.min()), oy + int(ys.max())
        bx0, bx1 = ox + int(xs.min()), ox + int(xs.max())
        crop = sub[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
        return crop, (by0, bx0), (by0, by1, bx0, bx1)

    if window is not None:
        h, w = img.shape
        wy0, wy1 = max(0, y - window), min(h, y + window + 1)
        wx0, wx1 = max(0, x - window), min(w, x + window + 1)
        labels, _ = ndimage.label(img[wy0:wy1, wx0:wx1] > level, structure=_EIGHT)
        rid = labels[y - wy0, x - wx0]
        if rid != 0:
            sub = labels == rid
            if not (
                sub[0].any() or sub[-1].any() or sub[:, 0].any() or sub[:, -1].any()
            ):
                return _pack(sub, wy0, wx0)
    labels, _ = ndimage.label(img > level, structure=_EIGHT)
    rid = labels[y, x]
    if rid == 0:  # cannot happen for a seed > threshold, but be explicit
        return np.ones((1, 1), bool), (y, x), (y, y, x, x)
    return _pack(labels == rid, 0, 0)


def _region_stats(
    region: np.ndarray, weights: np.ndarray, origin: tuple[int, int]
) -> tuple[float, float, float, float]:
    """(x, y, radius_px, aspect) of a bbox-cropped region from weighted moments.

    ``region`` is the bbox crop from :func:`_half_max_region`, ``origin`` its
    (y, x) position in the image and ``weights`` the full-image
    ``clip(img - bg, 0)``. ``aspect`` is the axis ratio
    ``sqrt(lambda_max / lambda_min)`` of the eigenvalues of the 2×2
    second-moment tensor ``[[Iyy, Ixy], [Ixy, Ixx]]``, so the major axis is
    always in the numerator regardless of the object's orientation — a
    horizontal hypha and a vertical one both report aspect > 1.
    """
    oy, ox = origin
    ys, xs = np.nonzero(region)
    ys = ys + oy
    xs = xs + ox
    w = weights[ys, xs]
    total = float(w.sum())
    if total <= 0:  # degenerate: fall back to the unweighted centroid
        w = np.ones(ys.size)
        total = float(ys.size)
    y = float((w * ys).sum() / total)
    x = float((w * xs).sum() / total)
    dy = ys - y
    dx = xs - x
    iyy = float((w * dy * dy).sum() / total)
    ixx = float((w * dx * dx).sum() / total)
    ixy = float((w * dy * dx).sum() / total)
    radius = math.sqrt(ys.size / math.pi)
    if iyy + ixx <= 0:  # single-pixel region: isotropic by definition
        return x, y, radius, 1.0
    lmin, lmax = (float(v) for v in np.linalg.eigvalsh([[iyy, ixy], [ixy, ixx]]))
    if lmax <= 0.0:
        return x, y, radius, 1.0
    if lmin <= 0.0:  # straight 1-px region: infinitely elongated, honestly
        return x, y, radius, math.inf
    return x, y, radius, math.sqrt(lmax / lmin)


def _regions_too_close(c: dict, u: dict) -> bool:
    """True if c's dilated region reaches u's region (the ``merged`` test).

    Both masks are stored bbox-cropped with their crop origins, so the AND
    runs only on the intersection of the bounding boxes — usually empty, so
    the pairwise gate stays cheap even for tiles with hundreds of candidates.
    """
    cy0, cy1, cx0, cx1 = c["dilated_bbox"]
    uy0, uy1, ux0, ux1 = u["bbox"]
    iy0, iy1 = max(cy0, uy0), min(cy1, uy1)
    ix0, ix1 = max(cx0, ux0), min(cx1, ux1)
    if iy0 > iy1 or ix0 > ix1:
        return False
    c_sub = c["dilated"][
        iy0 - c["d_origin"][0] : iy1 - c["d_origin"][0] + 1,
        ix0 - c["d_origin"][1] : ix1 - c["d_origin"][1] + 1,
    ]
    u_sub = u["region"][
        iy0 - u["origin"][0] : iy1 - u["origin"][0] + 1,
        ix0 - u["origin"][1] : ix1 - u["origin"][1] + 1,
    ]
    return bool((c_sub & u_sub).any())


def detect(img: np.ndarray, params: DetectionParams = DetectionParams()) -> list[Spot]:
    """Detect conidium-like spots in one tile; see the module docstring.

    Returns every candidate (accepted and rejected), ordered by descending
    ``score``. A flat image, or one whose noise is zero (``sigma_bg == 0``),
    yields ``[]``: without a noise estimate the ``k·sigma_bg`` threshold is
    undefined, and refusing to guess is the only honest answer.
    """
    img = np.asarray(img, dtype=np.float64)
    if img.ndim != 2 or img.size == 0:
        raise ValueError(f"detect: expected a non-empty 2-D image, got {img.shape}")
    bg, sigma = estimate_empty_background(img)
    if sigma <= 0.0:
        return []
    threshold = bg + params.thresholds * sigma

    seeds = (img == ndimage.maximum_filter(img, size=3, mode="nearest")) & (
        img > threshold
    )
    labels, n = ndimage.label(seeds, structure=_EIGHT)
    if n == 0:
        return []

    candidates = []
    weights = np.clip(img - bg, 0.0, None)
    for label_id in range(1, n + 1):
        ys, xs = np.nonzero(labels == label_id)
        k = int(np.argmax(img[ys, xs]))
        y, x = int(ys[k]), int(xs[k])
        peak = float(img[y, x])
        # Any region that could pass the size gate fits inside this window;
        # bigger objects fall back to the full-image labelling (see
        # _half_max_region) — rare, and they are rejected as too_large anyway.
        region, origin, bbox = _half_max_region(
            img, bg, y, x, window=int(4.0 * params.max_radius_px) + 2
        )
        cx, cy, radius, aspect = _region_stats(region, weights, origin)
        candidates.append(
            {
                "peak": peak,
                "score": (peak - bg) / sigma,
                "x": cx,
                "y": cy,
                "region": region,
                "origin": origin,
                "bbox": bbox,
                "radius_px": radius,
                "aspect": aspect,
            }
        )

    # Exact duplicates (two maxima on one plateau resolving to the same
    # centroid) collapse to the stronger one — they are one object.
    candidates.sort(key=lambda c: c["score"], reverse=True)
    unique: list[dict] = []
    for c in candidates:
        if any(
            math.hypot(c["x"] - u["x"], c["y"] - u["y"]) < 1e-9 for u in unique
        ):
            continue
        unique.append(c)

    # One dilation per candidate (not per pair). The 3×3 structure dilates a
    # set by exactly merge_r (Chebyshev) beyond its bbox, so the padded crop
    # below is truncation-free and the dilated bbox is exact without a scan.
    # The merged gate then costs only bbox-guarded ANDs on small crops.
    merge_r = int(math.ceil(params.merge_px))
    for c in unique:
        by0, by1, bx0, bx1 = c["bbox"]
        oy, ox = c["origin"]
        c["dilated"] = ndimage.binary_dilation(
            np.pad(c["region"], merge_r), iterations=merge_r, structure=_EIGHT
        )
        c["d_origin"] = (by0 - merge_r, bx0 - merge_r)
        c["dilated_bbox"] = (by0 - merge_r, by1 + merge_r,
                             bx0 - merge_r, bx1 + merge_r)

    spots: list[Spot] = []
    for c in unique:
        reason = ""
        if c["radius_px"] > params.max_radius_px:
            reason = "too_large"
        elif c["aspect"] > params.max_aspect:
            reason = "elongated"
        elif any(_regions_too_close(c, u) for u in unique if u is not c):
            reason = "merged"
        elif c["peak"] - bg < params.thresholds * sigma:
            # Unreachable for detect()'s own seeds (they are strictly above
            # the threshold); guards spot lists supplied by other code paths.
            reason = "too_dim"
        spots.append(
            Spot(
                x=c["x"],
                y=c["y"],
                score=c["score"],
                radius_px=c["radius_px"],
                aspect=c["aspect"],
                accepted=not reason,
                reason=reason,
            )
        )

    # Strongest first (the §8.1 confidence order).
    return spots


def load_manual(path) -> list[tuple[float, float]]:
    """Read a manual override: JSON list of ``[x, y]`` pairs (``(x, y)`` tuples).

    Seeds the detector's output for that tile — a manual list takes priority
    over any detection result (spec §8). Raises :class:`ValueError` when the
    file is not a JSON list of two-number pairs, so a typo cannot silently
    empty a tile's spot list.
    """
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise ValueError(f"load_manual: {path}: expected a JSON list of [x, y] pairs")
    out: list[tuple[float, float]] = []
    for i, item in enumerate(data):
        if (
            not isinstance(item, (list, tuple))
            or len(item) != 2
            or not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in item)
        ):
            raise ValueError(
                f"load_manual: {path}: entry {i} is not an [x, y] pair of numbers: {item!r}"
            )
        out.append((float(item[0]), float(item[1])))
    return out


def _add_tile_csv(rows: list[dict], csv_path: Path) -> None:
    """Append this run's rows to the summary CSV (header written once)."""
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with open(csv_path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_CSV_COLS)
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in _CSV_COLS})


def write_sanity_visuals(
    img: np.ndarray,
    spots: list[Spot],
    outdir,
    params: DetectionParams,
    tile_id: str,
) -> list[Path]:
    """Write the three §8.1 sanity artifacts for one tile; return their paths.

    ``spots_tile_<tile_id>.png`` — full-resolution overlay: accepted spots as
    green circles + id, rejected candidates as squares colour-coded by
    ``REASON_COLORS`` with the reason in the caption. An inset zooms on the
    strongest accepted spot (a full tile shrinks 1-px spots to nothing).

    ``spots_sheet.png`` — 48-px thumbnails upscaled ×4, ordered by score
    (spec §8.1: "ordered by confidence" — a colony outscoring the conidia
    lands at position 1, where the eyeball gate cannot miss it), all under
    ONE shared display window, accepted framed green / rejected orange with
    the reason (and the magnitude that failed the gate) in the caption.

    ``spots_summary.png`` + ``spots_summary.csv`` — per-tile counts,
    size/intensity histograms and the parameter set; the CSV is the full
    per-spot record including every rejected candidate and is APPENDED to,
    so repeated calls across a well form one run record. (A tile with zero
    candidates contributes no CSV rows — one row per candidate — but still
    appears with a 0/0 bar in the summary PNG of the call that wrote it.)
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    img = np.asarray(img, dtype=np.float64)
    if img.ndim != 2 or img.size == 0:
        raise ValueError(
            f"write_sanity_visuals: expected a non-empty 2-D image, got {img.shape}"
        )
    ranked = sorted(spots, key=lambda s: s.score, reverse=True)
    ids = {id(s): i + 1 for i, s in enumerate(ranked)}

    tile_png = outdir / _TILE_PNG.format(tile_id=tile_id)
    sheet_png = outdir / _SHEET_PNG
    summary_png = outdir / _SUMMARY_PNG
    csv_path = outdir / _SUMMARY_CSV

    # --- 1. tile overlay -------------------------------------------------
    _draw_tile_overlay(img, ranked, ids, params, tile_id, tile_png)

    # --- 2. contact sheet ------------------------------------------------
    _draw_contact_sheet(img, ranked, ids, tile_id, sheet_png)

    # --- 3. run summary (CSV + PNG) --------------------------------------
    rows = [
        {
            "tile_id": tile_id,
            "spot_id": ids[id(s)],
            "x": f"{s.x:.3f}",
            "y": f"{s.y:.3f}",
            "score": f"{s.score:.3f}",
            "radius_px": f"{s.radius_px:.3f}",
            "aspect": f"{s.aspect:.3f}",
            "accepted": s.accepted,
            "reason": s.reason,
        }
        for s in ranked
    ]
    _add_tile_csv(rows, csv_path)

    all_rows = _read_csv(csv_path)
    _draw_summary(all_rows, params, tile_id, summary_png)

    return [tile_png, sheet_png, summary_png, csv_path]


def _draw_tile_overlay(
    img: np.ndarray,
    ranked: list[Spot],
    ids: dict[int, int],
    params: DetectionParams,
    tile_id: str,
    out_png: Path,
) -> None:
    """Full tile + zoom inset, accepted green circles, rejected coloured squares."""
    h, w = img.shape
    lo, hi = float(np.percentile(img, 0.5)), float(np.percentile(img, 99.9))
    if hi <= lo:
        hi = lo + 1.0

    fig = plt.figure(figsize=(14, 7), dpi=110)
    grid = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.08)
    ax_full = fig.add_subplot(grid[0, 0])
    ax_zoom = fig.add_subplot(grid[0, 1])

    for ax in (ax_full, ax_zoom):
        ax.imshow(img, cmap="gray", vmin=lo, vmax=hi, interpolation="nearest")

    for s in ranked:
        if s.accepted:
            ax_full.add_patch(
                Circle((s.x, s.y), radius=max(2.0 * s.radius_px, 6.0), fill=False,
                       edgecolor="#2f9e44", linewidth=1.4)
            )
            ax_full.annotate(
                str(ids[id(s)]), (s.x, s.y), xytext=(4, -4),
                textcoords="offset points", color="#2f9e44", fontsize=7,
            )
        else:
            side = max(2.0 * s.radius_px + 6.0, 20.0)
            ax_full.add_patch(
                Rectangle((s.x - side / 2, s.y - side / 2), side, side, fill=False,
                          edgecolor=REASON_COLORS.get(s.reason, REASON_COLORS["other"]),
                          linewidth=1.4, linestyle="--")
            )

    ax_full.set_title(f"{tile_id}: {sum(s.accepted for s in ranked)} accepted, "
                      f"{sum(not s.accepted for s in ranked)} rejected")
    ax_full.set_xlabel("x [px]")
    ax_full.set_ylabel("y [px]")

    accepted = [s for s in ranked if s.accepted]
    if accepted:
        # Zoom on the STRONGEST accepted spot, not on the cluster centroid:
        # spots can be spread across the whole tile, and a centroid-centred
        # window would frame the empty space BETWEEN them.
        best = max(accepted, key=lambda s: s.score)
        half = 3.0 * params.max_radius_px  # a few PSF widths of context
        x0 = max(0.0, best.x - half)
        x1 = min(float(w), best.x + half)
        y0 = max(0.0, best.y - half)
        y1 = min(float(h), best.y + half)
        ax_zoom.set_xlim(x0, x1)
        ax_zoom.set_ylim(y1, y0)  # y increases downwards, as in image coordinates
        ax_zoom.set_title(
            f"zoom: strongest accepted spot #{ids[id(best)]} at "
            f"(x={best.x:.1f}, y={best.y:.1f}); {len(accepted)} accepted",
            fontsize=9,
        )
    else:
        ax_zoom.set_xlim(0, w)
        ax_zoom.set_ylim(h, 0)
        ax_zoom.set_title("no accepted spots", fontsize=9)
    ax_zoom.set_xticks([])
    ax_zoom.set_yticks([])

    legend_lines = ["● accepted (green circle, id)",
                    "■ too_large   (dark orange)",
                    "■ elongated   (amber)",
                    "■ merged      (magenta)",
                    "■ too_dim     (grey)"]
    fig.text(0.02, 0.02, "  |  ".join(legend_lines), fontsize=8, color="#333")
    fig.suptitle(
        f"Conidia detection sanity overlay — tile {tile_id}  "
        f"[k={params.thresholds}σ, R≤{params.max_radius_px}px, "
        f"aspect≤{params.max_aspect}, merge≤{params.merge_px}px]",
        fontsize=10,
    )
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def _thumbnail(img: np.ndarray, x: float, y: float, side: int) -> np.ndarray:
    """``side``×``side`` patch centred on (x, y), zero-padded past the border."""
    half = side // 2
    yc, xc = int(round(y)), int(round(x))
    out = np.zeros((side, side), np.float64)
    ys0, ys1 = max(0, yc - half), min(img.shape[0], yc - half + side)
    xs0, xs1 = max(0, xc - half), min(img.shape[1], xc - half + side)
    if ys0 < ys1 and xs0 < xs1:
        out[ys0 - (yc - half) : ys1 - (yc - half), xs0 - (xc - half) : xs1 - (xc - half)] = (
            img[ys0:ys1, xs0:xs1]
        )
    return out


def _caption(spot: Spot) -> str:
    """Gate-failure summary under each thumbnail (reason and the value that failed)."""
    if spot.accepted:
        return "accepted"
    if spot.reason == "too_large":
        return f"too_large r={spot.radius_px:.1f}px"
    if spot.reason == "elongated":
        return f"elongated aspect={spot.aspect:.1f}"
    return spot.reason or "rejected"


def _draw_contact_sheet(
    img: np.ndarray, ranked: list[Spot], ids: dict[int, int], tile_id: str, out_png: Path
) -> None:
    """48-px thumbnails ×4, ordered by score, green/orange frames + captions.

    ALL thumbnails share ONE display window (percentile stretch of the whole
    tile, the same doctrine as ``mosaic.panels_side_by_side``): per-thumbnail
    auto-normalisation would brighten a dim candidate and blow out a
    saturated one independently, so cross-candidate comparison — the whole
    point of the sheet — would be a display artifact.
    """
    rows = int(math.ceil(len(ranked) / _SHEET_COLS)) if ranked else 1
    cell = _SHEET_THUMB * _SHEET_ZOOM  # displayed size of one thumbnail, px
    top_px = 30  # suptitle strip
    bottom_px = 24  # keeps the last caption row inside the figure
    height_px = top_px + rows * (cell + _SHEET_CAPTION) + bottom_px
    fig = plt.figure(
        figsize=(_SHEET_COLS * cell / 100.0, height_px / 100.0), dpi=100
    )
    lo, hi = float(np.percentile(img, 0.5)), float(np.percentile(img, 99.9))
    if hi <= lo:
        hi = lo + 1.0
    for i, spot in enumerate(ranked):
        r = i // _SHEET_COLS
        ax = fig.add_axes(
            [
                (i % _SHEET_COLS) / _SHEET_COLS,
                1.0 - (top_px + r * (cell + _SHEET_CAPTION) + cell) / height_px,
                1.0 / _SHEET_COLS,
                cell / height_px,
            ]
        )
        ax.imshow(_thumbnail(img, spot.x, spot.y, _SHEET_THUMB), cmap="gray",
                  vmin=lo, vmax=hi, aspect="equal", interpolation="nearest")
        colour = "#2f9e44" if spot.accepted else REASON_COLORS.get(spot.reason, REASON_COLORS["other"])
        for spine in ax.spines.values():
            spine.set_edgecolor(colour)
            spine.set_linewidth(2.5)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.text(0.5, -0.10, f"#{ids[id(spot)]} {_caption(spot)}", transform=ax.transAxes,
                ha="center", va="top", fontsize=6, color=colour)
    fig.suptitle(
        f"Spot contact sheet — tile {tile_id}, {len(ranked)} candidates ordered by score "
        f"(green = accepted, orange = rejected with reason)", fontsize=9
    )
    fig.savefig(out_png)
    plt.close(fig)


def _read_csv(csv_path: Path) -> list[dict]:
    with open(csv_path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _draw_summary(
    all_rows: list[dict], params: DetectionParams, tile_id: str, out_png: Path,
) -> None:
    """Per-tile counts, size/intensity histograms for the whole run, parameters."""
    def accepted(rows):
        return [r for r in rows if r["accepted"] == "True"]

    tiles: list[str] = []
    for r in all_rows:  # insertion order = tile order of the run
        if r["tile_id"] not in tiles:
            tiles.append(r["tile_id"])
    if tile_id not in tiles:
        # A tile with zero candidates leaves no CSV rows, but it still ran —
        # a 0/0 bar is exactly what a dead or mis-normalized tile looks like.
        tiles.append(tile_id)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=100)
    ax_counts, ax_radius, ax_score, ax_text = axes.ravel()

    accepted_counts = [sum(r["accepted"] == "True" for r in all_rows if r["tile_id"] == t) for t in tiles]
    rejected_counts = [sum(r["accepted"] != "True" for r in all_rows if r["tile_id"] == t) for t in tiles]
    xs = np.arange(len(tiles))
    ax_counts.bar(xs, accepted_counts, color="#2f9e44", label="accepted")
    ax_counts.bar(xs, rejected_counts, bottom=accepted_counts, color="#e8590c", label="rejected")
    ax_counts.set_xticks(xs)
    ax_counts.set_xticklabels(tiles, rotation=90 if len(tiles) > 8 else 0, fontsize=7)
    ax_counts.set_ylabel("candidates")
    ax_counts.set_title(f"Per-tile counts — {len(accepted(all_rows))} accepted, "
                        f"{len(all_rows) - len(accepted(all_rows))} rejected")
    ax_counts.legend(fontsize=8)

    acc = accepted(all_rows)
    radius = [float(r["radius_px"]) for r in acc]
    score = [float(r["score"]) for r in acc]
    bins_r = np.linspace(0, max(1.0, params.max_radius_px), 25)
    ax_radius.hist(radius, bins=bins_r, color="#2f9e44")
    ax_radius.axvline(params.max_radius_px, color="#c92a2a", linestyle="--",
                      label=f"max_radius_px = {params.max_radius_px}")
    ax_radius.set_xlabel("equivalent radius [px] (accepted)")
    ax_radius.set_ylabel("spots")
    ax_radius.set_title("Size histogram")
    ax_radius.legend(fontsize=8)

    if score and max(score) > min(score):
        bins_s = np.linspace(min(min(score), params.thresholds), max(score), 25)
        ax_score.hist(score, bins=bins_s, color="#2f9e44")
    elif score:
        ax_score.hist(score, bins=[min(score) - 1.0, max(score) + 1.0], color="#2f9e44")
    ax_score.axvline(params.thresholds, color="#c92a2a", linestyle="--",
                     label=f"thresholds = {params.thresholds}σ_bg")
    ax_score.set_xlabel("score = (peak - bg) / sigma_bg (accepted)")
    ax_score.set_ylabel("spots")
    ax_score.set_title("Intensity histogram")
    ax_score.legend(fontsize=8)

    ax_text.axis("off")
    reason_counts = {r: sum(x["reason"] == r for x in all_rows) for r in REASONS}
    other = sum(
        x["reason"] not in REASONS and x["accepted"] != "True" for x in all_rows
    )
    lines = [
        f"Last tile: {tile_id}",
        "",
        "Parameter set",
        *[f"  {f.name} = {getattr(params, f.name)!r}" for f in fields(params)],
        "",
        "This run (all tiles in this directory)",
        f"  tiles: {len(tiles)}",
        f"  accepted: {len(acc)}",
        f"  rejected: {len(all_rows) - len(acc)}",
        *[f"    {r:<10} {reason_counts[r]}" for r in REASONS],
        *([f"    other      {other}"] if other else []),
        "",
        f"Record: {_SUMMARY_CSV} (one row per candidate, rejected included)",
        "",
        "Detection is used only at ≥ 20 accepted spots per well (spec §8).",
    ]
    ax_text.text(0.0, 1.0, "\n".join(lines), va="top", ha="left", family="monospace", fontsize=9)

    fig.suptitle("Conidia detection run summary (§8.1)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_png)
    plt.close(fig)