"""Whole-well mosaic assembly and side-by-side comparison figures (spec §2/§2.1).

The stitching phase (ashlar's aligner, wrapped by ``stitch_with_qc``) has
already estimated where every tile sits in the well; this module only
*places* tiles at those known offsets and builds the display figures the
pilot reports on. No re-registration happens here.

:func:`assemble` re-stitches one well from its tiles: float32 accumulator +
weight accumulator, each tile pasted at its (y, x) offset with a linear
feather ramp over ``blend_px`` pixels at its *internal* edges (edges that
lie on the mosaic border keep full weight). The final ``acc / max(w, eps)``
normalisation has a load-bearing consequence: a ramp on an edge that no
other tile overlaps is an exact no-op (``tile*w / w``), so the ramps only
shape genuine overlaps — where two tiles meet across exactly ``blend_px``
pixels their weights are complementary and sum to 1, giving a symmetric
linear crossfade with no brightness dip at the seam. The half-pixel offset
in the ramp keeps the edge pixel itself at positive weight, so a lone tile
reproduces exactly and no border line appears.

:func:`side_by_side` / :func:`panels_side_by_side` build the three-arm
comparison figures (raw / background-only / deconvolved, spec §2). ONE
display stretch is computed across ALL panels together and the same
``(x - lo) / (hi - lo)`` + gamma is applied to each, so differences between
panels are attributable to the processing, not to per-panel normalisation.
These PNGs are display-only 8-bit; the deconvolved mosaic itself is written
16-bit elsewhere.

:func:`read_ashlar_positions` reads tile positions from an ashlar-written
pyramidal OME-TIFF when the file carries them. Verified against real
ashlar v1.20 output: ``PyramidWriter`` does NOT store tile positions in the
file (the OME-XML is minimal, no StructuredAnnotations). The reader still
understands the layouts a writer may emit (a JSON blob with a
``mosaic``/``positions`` key in an OME XMLAnnotation or in the
ImageDescription); when none is present it raises
:class:`MosaicPositionsMissing` and the pilot runner passes the positions
recorded by ``stitch_with_qc`` instead. Either way the source used is
recorded in the log.
"""

from __future__ import annotations

import json
import logging
import xml.etree.ElementTree as ET

import numpy as np
import tifffile
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)

__all__ = [
    "assemble",
    "read_ashlar_positions",
    "side_by_side",
    "panels_side_by_side",
    "add_tile_labels",
    "MosaicPositionsMissing",
]

_SEPARATOR_PX = 6
_EPS = 1e-6


def _ramp(n: int, blend_px: int) -> np.ndarray:
    """Linear feather ramp for the first ``n`` pixels off an edge.

    Value at a pixel is its distance to the edge (pixel-centre convention,
    hence the half-pixel offset) over ``blend_px`` — the edge pixel keeps
    weight ``0.5 / blend_px > 0`` and the ramp reaches 1 exactly
    ``blend_px`` pixels in.
    """
    return np.clip((np.arange(n, dtype=np.float32) + 0.5) / blend_px, 0.0, 1.0)


def assemble(
    tiles: list[np.ndarray],
    positions: list[tuple[float, float]],
    out_shape: tuple[int, int],
    blend_px: int = 64,
) -> np.ndarray:
    """Place tiles at known (y, x) pixel offsets into one float32 mosaic.

    Placement only — offsets come from the stitching phase and are never
    re-estimated here. Overlaps are feather-blended: each tile carries full
    weight except over ``blend_px`` pixels at its internal edges, where the
    weight ramps linearly to the edge. Tiles may extend past the canvas
    (clipped) and may be different sizes; positions are rounded to integer
    pixels.
    """
    if len(tiles) != len(positions):
        raise ValueError(
            f"got {len(tiles)} tiles but {len(positions)} positions"
        )

    out_h, out_w = int(out_shape[0]), int(out_shape[1])
    acc = np.zeros((out_h, out_w), np.float32)
    wacc = np.zeros((out_h, out_w), np.float32)

    for tile, (y0, x0) in zip(tiles, positions):
        y0, x0 = int(round(float(y0))), int(round(float(x0)))
        tile32 = np.asarray(tile, np.float32)
        th, tw = tile32.shape
        ys0, ys1 = max(y0, 0), min(y0 + th, out_h)
        xs0, xs1 = max(x0, 0), min(x0 + tw, out_w)
        if ys0 >= ys1 or xs0 >= xs1:
            continue  # entirely off-canvas
        sub = tile32[ys0 - y0 : ys1 - y0, xs0 - x0 : xs1 - x0]
        w = np.ones(sub.shape, np.float32)
        if blend_px > 0:
            # Ramps only at internal edges: an edge on the mosaic border
            # (or beyond it, after clipping) keeps full weight.
            if y0 > 0:
                n = min(blend_px, sub.shape[0])
                w[:n] *= _ramp(n, blend_px)[:, None]
            if y0 + th < out_h:
                n = min(blend_px, sub.shape[0])
                w[-n:] *= _ramp(n, blend_px)[::-1][:, None]
            if x0 > 0:
                n = min(blend_px, sub.shape[1])
                w[:, :n] *= _ramp(n, blend_px)[None, :]
            if x0 + tw < out_w:
                n = min(blend_px, sub.shape[1])
                w[:, -n:] *= _ramp(n, blend_px)[::-1][None, :]
        acc[ys0:ys1, xs0:xs1] += sub * w
        wacc[ys0:ys1, xs0:xs1] += w

    return acc / np.maximum(wacc, _EPS)


class MosaicPositionsMissing(LookupError):
    """The mosaic file carries no tile positions in its metadata."""


def _find_element(root: ET.Element, localname: str) -> ET.Element | None:
    """First element with the given local name, whatever the XML namespace."""
    for el in root.iter():
        if el.tag.rsplit("}", 1)[-1] == localname:
            return el
    return None


def _fullres_shape(tf: tifffile.TiffFile) -> tuple[int, int]:
    """(height, width) of the full-resolution mosaic level."""
    xml = tf.ome_metadata
    if xml:
        try:
            pix = _find_element(ET.fromstring(xml), "Pixels")
        except ET.ParseError:
            pix = None
        if pix is not None and "SizeY" in pix.attrib and "SizeX" in pix.attrib:
            return int(pix.attrib["SizeY"]), int(pix.attrib["SizeX"])
    series = tf.series[0]
    axes = series.axes
    h = series.shape[axes.index("Y")] if "Y" in axes else series.shape[-2]
    w = series.shape[axes.index("X")] if "X" in axes else series.shape[-1]
    return int(h), int(w)


def _candidate_json_strings(tf: tifffile.TiffFile) -> list[str]:
    """Metadata strings that might carry a positions JSON blob."""
    out: list[str] = []
    xml = tf.ome_metadata
    if xml and "StructuredAnnotations" in xml:
        try:
            root = ET.fromstring(xml)
        except ET.ParseError:
            root = None
        if root is not None:
            for el in root.iter():
                if el.tag.rsplit("}", 1)[-1] in ("Value", "Text"):
                    text = "".join(el.itertext()).strip()
                    if text:
                        out.append(text)
    desc = tf.pages[0].tags.get("ImageDescription")
    if desc is not None:
        d = desc.value
        if isinstance(d, bytes):
            d = d.decode("utf-8", "replace")
        d = d.strip()
        if d.startswith("{"):
            out.append(d)
    return out


def _positions_from_json(obj) -> list[tuple[float, float]] | None:
    """Extract [(y, x), ...] tile positions from parsed JSON, or None."""
    if isinstance(obj, dict):
        for key in ("mosaic", "positions"):
            if key in obj:
                obj = obj[key]
                break
        else:
            return None
    if not isinstance(obj, (list, tuple)) or not obj:
        return None
    pos = []
    for p in obj:
        if not isinstance(p, (list, tuple)) or len(p) != 2:
            return None
        pos.append((float(p[0]), float(p[1])))
    return pos


def read_ashlar_positions(
    mosaic_path,
) -> tuple[list[tuple[float, float]], tuple[int, int]]:
    """Read tile positions and mosaic size from an ashlar pyramidal OME-TIFF.

    Returns ``((y, x) offsets in pixels, (height, width))`` — the same
    position convention :func:`assemble` takes. The full-resolution mosaic
    level defines the shape (SizeY/SizeX of the OME Pixels element).

    If the file carries no positions (as real ashlar v1.20 output does not),
    raises :class:`MosaicPositionsMissing`; the pilot runner then passes the
    positions recorded by ``stitch_with_qc`` instead. Which source was used
    is recorded in the log either way.
    """
    path = str(mosaic_path)
    with tifffile.TiffFile(path) as tf:
        shape = _fullres_shape(tf)
        for text in _candidate_json_strings(tf):
            try:
                obj = json.loads(text)
            except (json.JSONDecodeError, ValueError):
                continue
            positions = _positions_from_json(obj)
            if positions is not None:
                logger.info(
                    "read_ashlar_positions: %s — using the %d tile positions "
                    "from the mosaic metadata",
                    path,
                    len(positions),
                )
                return positions, shape
    logger.info(
        "read_ashlar_positions: %s — no tile positions in the mosaic "
        "metadata; caller must pass positions from stitch_with_qc",
        path,
    )
    raise MosaicPositionsMissing(
        f"{path}: no tile positions in the mosaic metadata — pass positions "
        "from stitch_with_qc instead"
    )


def _stretch_value(spec, pixels: np.ndarray) -> float:
    """A stretch endpoint given as ``"p<percentile>"`` or a raw value."""
    if isinstance(spec, str):
        if spec.startswith("p"):
            return float(np.percentile(pixels, float(spec[1:])))
        return float(spec)
    return float(spec)


def _shared_stretch(
    arrs: list[np.ndarray], stretch: tuple
) -> tuple[float, float]:
    """ONE (lo, hi) display window computed across ALL panels together."""
    pixels = np.concatenate([a.ravel() for a in arrs])
    lo = _stretch_value(stretch[0], pixels)
    hi = _stretch_value(stretch[1], pixels)
    if not np.isfinite(lo):
        lo = 0.0
    if not np.isfinite(hi) or hi <= lo:
        hi = lo + 1.0  # constant window guard: keep the mapping defined
    return lo, hi


def _to_uint8(a: np.ndarray, lo: float, hi: float, gamma: float) -> np.ndarray:
    x = (a - lo) / (hi - lo)
    np.clip(x, 0.0, 1.0, out=x)
    if gamma != 1.0:
        x = np.power(x, gamma, dtype=np.float32)
    return (x * 255.0 + 0.5).astype(np.uint8)


def panels_side_by_side(
    panels: list[np.ndarray],
    out_png,
    stretch: tuple = ("p1", "p99.7"),
    gamma: float = 0.85,
) -> None:
    """Write N panels side by side under ONE shared display stretch.

    The (lo, hi) window is computed from the pixel population of all panels
    combined and the same normalisation + gamma is applied to each panel, so
    panel-to-panel differences survive normalisation (three-arm comparison,
    spec §2). Panels may differ in size; smaller ones are top-aligned and
    zero-padded. A black separator strip keeps panel pixels apart.
    """
    if not panels:
        raise ValueError("need at least one panel")
    arrs = [np.asarray(p, np.float32) for p in panels]
    lo, hi = _shared_stretch(arrs, stretch)
    rendered = [_to_uint8(a, lo, hi, gamma) for a in arrs]

    height = max(r.shape[0] for r in rendered)
    row: list[np.ndarray] = []
    for i, r in enumerate(rendered):
        if i:
            row.append(np.zeros((height, _SEPARATOR_PX), np.uint8))
        if r.shape[0] < height:
            r = np.pad(r, ((0, height - r.shape[0]), (0, 0)))
        row.append(r)
    img = row[0] if len(row) == 1 else np.concatenate(row, axis=1)
    Image.fromarray(img).save(str(out_png))


def side_by_side(
    before: np.ndarray,
    after: np.ndarray,
    out_png,
    stretch: tuple = ("p1", "p99.7"),
    gamma: float = 0.85,
) -> None:
    """Raw-vs-processed comparison: two panels, one shared stretch."""
    panels_side_by_side([before, after], out_png, stretch=stretch, gamma=gamma)


def add_tile_labels(png_path, positions, labels, margin: int = 28) -> None:
    """Draw per-tile labels (e.g. dz values) in a margin strip above the image.

    Opens the PNG at ``png_path``, expands it with a black top strip of
    ``margin`` pixels, writes one label per tile at the tile's x offset
    (positions are (y, x) mosaic pixel offsets, as in :func:`assemble`), and
    saves back to the same path.
    """
    if len(positions) != len(labels):
        raise ValueError(
            f"got {len(positions)} positions but {len(labels)} labels"
        )
    img = Image.open(png_path)
    width, height = img.size
    canvas = Image.new(img.mode, (width, height + margin), 0)
    canvas.paste(img, (0, margin))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=16)
    for (_, x), label in zip(positions, labels):
        draw.text((int(round(float(x))) + 4, 6), str(label), fill=255, font=font)
    canvas.save(str(png_path))
