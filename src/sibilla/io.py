"""CZI / TIFF I/O for real data (spec §3 io.py row, probe 2026-10-19).

- CZI metadata -> RealMeta (voxel, shape, channels, objective).
- Channel loads and 3D crop cuts from the full CZI.
- NCC-based crop location (spec §1: crops are exact sub-block extracts).
- ImageJ-style crop TIFF reading.

czifile 2026.10.14 API notes (probe-verified): `metadata()` is a method;
`asarray()` returns (C, Z, Y, X). One channel of this CZI is ~1.08 GB.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tifffile


@dataclass(frozen=True)
class ChannelInfo:
    index: int
    name: str
    dye: str
    excitation_nm: float | None
    emission_nm: float | None


@dataclass(frozen=True)
class RealMeta:
    voxel_um: tuple[float, float, float]      # (x, y, z)
    shape: tuple[int, int, int, int]          # (C, Z, Y, X)
    channels: tuple[ChannelInfo, ...]
    na: float
    n_im: float
    magnification: float
    objective_name: str


def _tag(el: ET.Element) -> str:
    return el.tag.split("}")[-1]


def _find_all(root: ET.Element, name: str) -> list[ET.Element]:
    return [e for e in root.iter() if _tag(e) == name]


def _text(el: ET.Element | None) -> str | None:
    return el.text.strip() if el is not None and el.text else None


def _f(el: ET.Element | None) -> float | None:
    t = _text(el)
    return float(t) if t is not None else None


def _parse_metadata_xml(xml: str) -> RealMeta:
    """Parse a CZI Metadata XML string into RealMeta."""
    root = ET.fromstring(xml)

    # --- voxel scaling
    dist: dict[str, float] = {}
    for d in _find_all(root, "Distance"):
        v = _f(d.find("Value"))
        if d.get("Id") in ("X", "Y", "Z") and v is not None:
            dist[d.get("Id")] = v * 1e6          # m -> µm
    if not {"X", "Y", "Z"} <= dist.keys():
        raise ValueError("CZI metadata: missing X/Y/Z scaling")
    voxel = (dist["X"], dist["Y"], dist["Z"])

    # --- channels
    channels: list[ChannelInfo] = []
    seen: set[int] = set()
    for ch in _find_all(root, "Channel"):
        idx_s = ch.get("Id", "")
        m = re.search(r"(\d+)$", idx_s)
        if not m or ch.get("Name") is None:
            continue
        idx = int(m.group(1))
        if idx in seen:
            continue
        seen.add(idx)
        dye = _text(ch.find("DyeName")) or ""
        channels.append(ChannelInfo(
            index=idx,
            name=ch.get("Name", ""),
            dye=dye,
            excitation_nm=_f(ch.find("ExcitationWavelength")),
            emission_nm=_f(ch.find("EmissionWavelength")),
        ))
    channels.sort(key=lambda c: c.index)

    # --- shape: SizeX/Y/Z/C of the first Image branch (main document)
    size = {}
    for tag_name, key in (("SizeX", "x"), ("SizeY", "y"),
                          ("SizeZ", "z"), ("SizeC", "c")):
        els = _find_all(root, tag_name)
        if els:
            size[key] = int(_text(els[0]))
    shape = (size.get("c", 0) or len(channels), size.get("z", 0),
             size.get("y", 0), size.get("x", 0))

    # --- objective: the experiment's reference must resolve to a block
    ref_id = None
    for os_el in _find_all(root, "ObjectiveSettings"):
        for e in os_el.iter():
            if _tag(e) == "ObjectiveRef" and e.get("Id"):
                ref_id = e.get("Id")
                break
        if ref_id:
            break
    if ref_id is None:
        raise ValueError("CZI metadata: no ObjectiveSettings/ObjectiveRef Id")
    obj_block = None
    for ob in _find_all(root, "Objective"):
        if ob.get("Id") == ref_id and ob.find("LensNA") is not None:
            obj_block = ob
            break
    if obj_block is None:
        raise ValueError(f"CZI metadata: objective Ref {ref_id!r} unresolved")

    na = _f(obj_block.find("LensNA"))
    n_im = _f(obj_block.find("ImmersionRefractiveIndex"))
    mag = _f(obj_block.find("NominalMagnification"))
    if na is None or n_im is None or mag is None:
        raise ValueError("CZI metadata: objective block missing NA/n/mag")

    return RealMeta(
        voxel_um=voxel,
        shape=shape,
        channels=tuple(channels),
        na=na,
        n_im=n_im,
        magnification=mag,
        objective_name=obj_block.get("Name", ""),
    )


def read_czi_metadata(path: str | Path) -> RealMeta:
    """RealMeta from a CZI file (czifile 2026.10.14: metadata() is a method)."""
    import czifile

    with czifile.CziFile(str(path)) as czi:
        xml = czi.metadata()
    return _parse_metadata_xml(xml)


def load_channel(path: str | Path, channel: int) -> np.ndarray:
    """Load one channel as (Z, Y, X) float32 (memory-safe: one channel)."""
    import czifile

    with czifile.CziFile(str(path)) as czi:
        arr = czi.asarray()
    a = np.asarray(arr)
    if a.ndim != 4:
        raise ValueError(f"expected (C, Z, Y, X), got {a.shape}")
    if not (0 <= channel < a.shape[0]):
        raise ValueError(f"channel {channel} out of range 0..{a.shape[0]-1}")
    return a[channel].astype(np.float32)


def cut_crop_3d(path: str | Path, channel: int, z0: int, z1: int,
                y0: int, x0: int, h: int, w: int) -> np.ndarray:
    """Cut a 3D crop (Nz, H, W) float32, z0..z1 inclusive, from the CZI."""
    vol = load_channel(path, channel)
    nz = vol.shape[0]
    if not (0 <= z0 <= z1 < nz):
        raise ValueError(f"z range [{z0},{z1}] outside 0..{nz-1}")
    return vol[z0:z1 + 1, y0:y0 + h, x0:x0 + w].astype(np.float32)


def square_pad(stack: np.ndarray,
               value: float | np.ndarray | None = None) -> np.ndarray:
    """Pad (Nz, M, N) to (Nz, S, S) with S = max(M, N).

    value=None pads with the per-plane median (spec: fitting operates on
    square grids per build_psf_stack's convention; per-plane median keeps
    the pad background-neutral for the per-plane background columns).
    """
    nz, m, n = stack.shape
    s = max(m, n)
    if m == s and n == s:
        return stack.astype(np.float32, copy=True)
    if value is None:
        pad = np.median(stack, axis=(1, 2)).astype(np.float32)
    else:
        pad = np.asarray(value, dtype=np.float32)
    out = np.empty((nz, s, s), dtype=np.float32)
    for k in range(nz):
        out[k] = pad[k] if np.ndim(pad) else pad
    out[:, :m, :n] = stack
    return out


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    p = a - a.mean()
    q = b - b.mean()
    d = np.sqrt((p * p).sum() * (q * q).sum())
    return float((p * q).sum() / d) if d > 0 else 0.0


def locate_stack(volume: np.ndarray, template_plane: np.ndarray) -> tuple[float, int, int]:
    """Locate a template plane in a volume plane by FFT cross-correlation.

    Returns (ncc, y0, x0) — the NCC of the best-matching window and its
    top-left corner. NCC < ~0.8 means "not this location/channel".
    """
    from scipy.signal import fftconvolve

    best: tuple[float, int, int] = (-2.0, 0, 0)
    H, W = template_plane.shape
    ts = template_plane.astype(np.float64) - template_plane.mean()
    ts_e = np.sqrt((ts * ts).sum())
    if ts_e == 0:
        return 0.0, 0, 0
    for z in range(volume.shape[0]):
        im = volume[z].astype(np.float64)
        imc = im - im.mean()
        if im.shape[0] < H or im.shape[1] < W:
            continue
        cc = fftconvolve(imc, ts[::-1, ::-1], mode="valid")
        if cc.size == 0:
            continue
        iy, ix = np.unravel_index(np.argmax(cc), cc.shape)
        patch = imc[iy:iy + H, ix:ix + W]
        v = float((patch * ts).sum() / np.sqrt((patch * patch).sum()) / ts_e)
        if v > best[0]:
            best = (v, int(iy), int(ix))
    return best


def locate_z_band(volume: np.ndarray, stack: np.ndarray,
                  y0: int, x0: int) -> tuple[float, int]:
    """Best-matching z-window for a stack at fixed lateral offset.

    Returns (ncc, z0) with z0 the first plane of the best window.
    """
    nz_c = stack.shape[0]
    H, W = stack.shape[1:3]
    best: tuple[float, int] = (-2.0, 0)
    for z0 in range(volume.shape[0] - nz_c + 1):
        win = volume[z0:z0 + nz_c, y0:y0 + H, x0:x0 + W].astype(np.float64)
        v = _ncc(win, stack.astype(np.float64))
        if v > best[0]:
            best = (v, z0)
    return best


def read_crop_tiff(path: str | Path) -> tuple[np.ndarray, dict]:
    """Read an ImageJ-style crop TIFF -> (stack float32, tags dict)."""
    with tifffile.TiffFile(str(path)) as tif:
        stack = tif.series[0].asarray().astype(np.float32)
        tags = {}
        pg = tif.pages[0]
        if "XResolution" in pg.tags:
            num, den = pg.tags["XResolution"].value
            tags["pixel_size_um"] = float(den) / float(num)      # µm per px
        desc = pg.tags.get("ImageDescription")
        if desc is not None and "spacing" in str(desc.value):
            m = re.search(r"spacing=([0-9.]+)", str(desc.value))
            if m:
                tags["z_step_um"] = float(m.group(1))
    return stack, tags
