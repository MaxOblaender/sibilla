"""Random tubule-network scene: generation + supersampled rendering (plan §2).

Conventions:

- Scene is planar: every tubule lies in the object plane (z = plane 0 of the
  z_planes_um grid; per-tubule z-jitter is off by default, plan §2).
- Coordinates: (x, y) in pixels with x = column, y = row.
- Segments are straight lines with endpoints (x0, y0, x1, y1); lengths
  uniform in [4, 7] µm; angles uniform in [0, π).
- Rejection sampling (plan §2):
    * minimum segment–segment distance >= min_sep_um (1.0 µm)
    * border margin >= border_margin_um (1.0 µm)
    * no crossings in v1
    * orientation coverage: redraw the whole set if the largest angular gap
      between segment orientations (mod π) exceeds 100°
    * fallback if acceptance collapses: relax to 0.8 µm, then 8 segments
- Rendering: each tubule is a filled circle of the *fluorescent* radius
  (diameter 60 nm by default) swept along its segment, drawn on an 8×
  supersampled grid (60 nm px) and downsampled by 8×8 box binning — the
  binning simultaneously models detector pixel integration (plan §2).
  The resulting object O_s has integral == 1 (unit photons), so
  `amp_s * B_nom * (O_s ⊛ PSF_j)` is the per-segment photon model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .defaults import DEFAULT_SCENE, SceneSpec


@dataclass
class Scene:
    spec: SceneSpec
    segments: np.ndarray                       # (S, 4) [x0, y0, x1, y1] px
    angles_rad: np.ndarray                     # (S,)
    objects: list[np.ndarray] = field(default_factory=list)   # (S) unit-photon (M, N)

    @property
    def n_segments(self) -> int:
        return len(self.segments)

    @property
    def length_um(self) -> np.ndarray:
        dx = (self.segments[:, 2] - self.segments[:, 0]) * self.spec.pixel_size_um
        dy = (self.segments[:, 3] - self.segments[:, 1]) * self.spec.pixel_size_um
        return np.hypot(dx, dy)


def _point_segment_dist_um(p: np.ndarray, seg: np.ndarray,
                           spec: SceneSpec) -> float:
    """Minimum distance in µm from point p=[x,y] px to segment seg=[x0,y0,x1,y1] px."""
    ax, ay, bx, by = seg
    vx, vy = bx - ax, by - ay
    L2 = vx * vx + vy * vy
    if L2 < 1e-12:
        return float(np.hypot(p[0] - ax, p[1] - ay)) * spec.pixel_size_um
    t = ((p[0] - ax) * vx + (p[1] - ay) * vy) / L2
    t = min(1.0, max(0.0, t))
    qx, qy = ax + t * vx, ay + t * vy
    return float(np.hypot(p[0] - qx, p[1] - qy)) * spec.pixel_size_um


def _segments_min_dist_um(a: np.ndarray, b: np.ndarray,
                          spec: SceneSpec) -> float:
    """Minimum segment–segment distance (µm); < ~0 if they cross."""
    d = min(
        _point_segment_dist_um(a[:2], b, spec),
        _point_segment_dist_um(a[2:], b, spec),
        _point_segment_dist_um(b[:2], a, spec),
        _point_segment_dist_um(b[2:], a, spec),
    )
    # crossing detection: endpoints on opposite sides of each other's line
    def cross(ax, ay, bx, by, cx, cy):
        return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
    x0, y0, x1, y1 = a
    u0, v0, u1, v1 = b
    c1 = cross(x0, y0, x1, y1, u0, v0)
    c2 = cross(x0, y0, x1, y1, u1, v1)
    c3 = cross(u0, v0, u1, v1, x0, y0)
    c4 = cross(u0, v0, u1, v1, x1, y1)
    if ((c1 > 0) != (c2 > 0)) and ((c3 > 0) != (c4 > 0)):
        d = min(d, 0.0)
    return d


def _max_angular_gap_deg(angles: np.ndarray) -> float:
    """Largest gap between orientations (mod π), in degrees."""
    a = np.mod(angles, np.pi)
    a = np.sort(a)
    gaps = np.diff(a)
    gaps = np.concatenate([gaps, [np.pi - a[-1] + a[0]]])
    return float(np.max(gaps) * 180.0 / np.pi)


def _generate_segments(spec: SceneSpec, rng: np.random.Generator,
                       min_sep_um: float, n_segments: int) -> tuple[np.ndarray, np.ndarray]:
    fov = spec.fov_px
    px_um = spec.pixel_size_um
    margin_px = spec.border_margin_um / px_um
    min_sep_px = min_sep_um / px_um
    lo_px = (spec.length_range_um[0]) / px_um
    hi_px = (spec.length_range_um[1]) / px_um

    segs: list[np.ndarray] = []
    angles: list[float] = []
    attempts = 0
    while len(segs) < n_segments and attempts < spec.max_attempts:
        attempts += 1
        length = rng.uniform(lo_px, hi_px)
        ang = rng.uniform(0.0, np.pi)
        # center uniformly in the allowed box, then check border margin
        cx = rng.uniform(margin_px, fov - margin_px)
        cy = rng.uniform(margin_px, fov - margin_px)
        dx, dy = 0.5 * length * np.cos(ang), 0.5 * length * np.sin(ang)
        x0, y0, x1, y1 = cx - dx, cy - dy, cx + dx, cy + dy
        seg = np.array([x0, y0, x1, y1])
        # border margin: whole segment inside [margin, fov-margin]
        if (seg.min() < margin_px or seg.max() > fov - margin_px):
            continue
        ok = True
        for s in segs:
            if _segments_min_dist_um(seg, s, spec) < min_sep_px * px_um:
                ok = False
                break
        if not ok:
            continue
        segs.append(seg)
        angles.append(ang)
    segs_arr = np.asarray(segs, dtype=float)
    ang_arr = np.asarray(angles, dtype=float)
    return segs_arr, ang_arr


def generate_scene(spec: SceneSpec = DEFAULT_SCENE,
                   seed: int = 0) -> Scene:
    """Generate the tubule network with the plan §2 rejection rules."""
    rng = np.random.default_rng(seed)
    segs, angs = _generate_segments(spec, rng, spec.min_sep_um, spec.n_segments)
    used_fallback_sep = False
    if len(segs) < spec.n_segments:
        # fallback step 1: relax separation to 0.8 µm
        segs, angs = _generate_segments(spec, rng, spec.fallback_min_sep_um, spec.n_segments)
        used_fallback_sep = True
    if len(segs) < spec.n_segments:
        # fallback step 2: drop to 8 segments (plan §2)
        spec = SceneSpec(**{**spec.__dict__, "n_segments": spec.fallback_n_segments})
        segs, angs = _generate_segments(spec, rng, spec.min_sep_um, spec.n_segments)
    if len(segs) < spec.n_segments:
        raise RuntimeError(
            f"scene generation failed: only {len(segs)}/{spec.n_segments} segments after "
            f"fallbacks (attempts={spec.max_attempts})")

    # orientation coverage: redraw the whole set if the angular gap is too large.
    # A redraw is accepted ONLY if it keeps every segment — a failed redraw
    # would silently drop tubules from the scene (plan §2 forbids partial sets).
    for _ in range(50):
        if _max_angular_gap_deg(angs) <= spec.max_angular_gap_deg:
            break
        new_segs, new_angs = _generate_segments(spec, rng,
                                                spec.fallback_min_sep_um if used_fallback_sep
                                                else spec.min_sep_um,
                                                len(segs))
        if len(new_segs) == len(segs):
            segs, angs = new_segs, new_angs

    if len(segs) < spec.n_segments:
        raise RuntimeError(
            f"scene generation produced only {len(segs)}/{spec.n_segments} "
            f"segments (seed={seed}); try another scene_seed")

    scene = Scene(spec=spec, segments=segs, angles_rad=angs)
    scene.objects = [render_tubule_unit(*seg, spec) for seg in segs]
    return scene


def render_tubule_unit(x0: float, y0: float, x1: float, y1: float,
                       spec: SceneSpec | None = None,
                       diameter_um: float | None = None) -> np.ndarray:
    """Unit-photon tubule object: supersampled sweep + 8×8 box binning.

    x0..y1 are segment endpoints in detector pixels. The tubule radius is
    `diameter_um / 2` (default: spec.fluorescent_diameter_um / 2). The
    returned (M, N) array has integral exactly 1 (sum == number of
    super-pixels inside the swept region, divided by that number).
    """
    spec = spec or DEFAULT_SCENE
    s = spec.supersample
    px_um = spec.pixel_size_um
    d = diameter_um if diameter_um is not None else spec.fluorescent_diameter_um
    m, n = spec.fov_px, spec.fov_px
    S = s * m
    # superpixel grid: a detector pixel covers s×s superpixels; pixel (i, j)
    # center is at ((i + 0.5) * s) in superpixel coords.
    ys, xs = np.mgrid[0:S, 0:S].astype(float)
    # work in superpixel units; segment endpoints -> superpixel coords of px centers
    ax, ay = (x0 + 0.5) * s, (y0 + 0.5) * s
    bx, by = (x1 + 0.5) * s, (y1 + 0.5) * s
    vx, vy = bx - ax, by - ay
    L2 = vx * vx + vy * vy
    if L2 < 1e-12:
        t = np.zeros((S, S))
    else:
        t = ((xs - ax) * vx + (ys - ay) * vy) / L2
        t = np.clip(t, 0.0, 1.0)
    qx = ax + t * vx
    qy = ay + t * vy
    dist = np.hypot(xs - qx, ys - qy)
    radius_sp = (d / 2.0) / px_um * s
    filled = dist <= radius_sp
    count = int(filled.sum())
    if count == 0:
        # degenerate: put one superpixel at the segment midpoint
        i = int(np.clip(int(0.5 * (ay + by)), 0, S - 1))
        j = int(np.clip(int(0.5 * (ax + bx)), 0, S - 1))
        filled[i, j] = True
        count = 1
    img = filled.astype(float) / count
    # 8×8 box binning -> detector grid (models pixel integration)
    img = img.reshape(m, s, n, s).sum(axis=(1, 3))
    return img


def render_scene_unit(scene: Scene, diameter_um: float | None = None) -> list[np.ndarray]:
    """Re-render every segment as a unit-photon object (with an optional
    diameter override — used for the wrong-anchor experiment)."""
    return [render_tubule_unit(*seg, scene.spec, diameter_um=diameter_um)
            for seg in scene.segments]


# --- 3D capsule object model (spec §3 scene.py row, real-data plan Task 4) ----


@dataclass(frozen=True)
class CapsuleSpec:
    """Grid + object geometry for 3D capsule rendering.

    Endpoints are in detector pixels (xy) and PLANE units (z).
    """
    fov_px_y: int
    fov_px_x: int
    n_planes: int
    pixel_size_um: float
    z_step_um: float
    diameter_um: float
    aa_width_px: float = 1.0          # anti-alias edge ramp width, pixels


@dataclass
class CapsuleScene:
    spec: CapsuleSpec
    segments: np.ndarray              # (S, 6) [x0, y0, z0, x1, y1, z1]
    angles_rad: np.ndarray            # (S,) in-plane angle
    tilts_deg: np.ndarray             # (S,) tilt out of the xy-plane
    objects: list[np.ndarray] = field(default_factory=list)   # unit-sum (Nz, M, N)


def render_capsule_unit(x0: float, y0: float, z0: float,
                        x1: float, y1: float, z1: float,
                        spec: CapsuleSpec) -> np.ndarray:
    """Unit-sum 3D capsule footprint on the (Nz, M, N) grid.

    Analytic distance-to-segment coverage with a linear anti-alias ramp of
    width `aa_width_px` pixels at the capsule edge (sub-pixel accurate, no
    supersampling):
        coverage = clip(0.5 − (dist_µm − r) / (aa_width_px · px), 0, 1)
    """
    nz, m, n = spec.n_planes, spec.fov_px_y, spec.fov_px_x
    px = spec.pixel_size_um
    # voxel-center coordinates in µm, origin at voxel [0, 0, 0] center
    xs = np.arange(n) * px
    ys = np.arange(m) * px
    zs = np.arange(nz) * spec.z_step_um
    # segment endpoints in µm (detector pixel -> µm via pixel pitch)
    ax, ay, az = x0 * px, y0 * px, z0 * spec.z_step_um
    bx, by, bz = x1 * px, y1 * px, z1 * spec.z_step_um
    vx, vy, vz = bx - ax, by - ay, bz - az
    L2 = vx * vx + vy * vy + vz * vz

    # broadcast: (nz, m, n) grids of µm coordinates
    Z = zs[:, None, None]
    Y = ys[None, :, None]
    X = xs[None, None, :]
    if L2 < 1e-12:
        qx, qy, qz = X * 0 + ax, Y * 0 + ay, Z * 0 + az
    else:
        t = ((X - ax) * vx + (Y - ay) * vy + (Z - az) * vz) / L2
        t = np.clip(t, 0.0, 1.0)
        qx = ax + t * vx
        qy = ay + t * vy
        qz = az + t * vz
    dist = np.sqrt((X - qx) ** 2 + (Y - qy) ** 2 + (Z - qz) ** 2)
    r = spec.diameter_um / 2.0
    ramp = spec.aa_width_px * px
    cov = np.clip(0.5 - (dist - r) / ramp, 0.0, 1.0)
    s = cov.sum()
    if s <= 0:
        # degenerate: single voxel at the segment midpoint
        out = np.zeros((nz, m, n))
        ik = int(np.clip(round((z0 + z1) / 2), 0, nz - 1))
        iy = int(np.clip(round((y0 + y1) / 2), 0, m - 1))
        ix = int(np.clip(round((x0 + x1) / 2), 0, n - 1))
        out[ik, iy, ix] = 1.0
        return out
    return cov / s


def generate_capsule_scene(spec: CapsuleSpec, seed: int, n_segments: int = 10,
                           length_range_um: tuple[float, float] = (1.0, 3.0),
                           z_center_range_um: tuple[float, float] = (-1.0, 1.0),
                           max_tilt_deg: float = 0.0,
                           min_sep_um: float = 1.5) -> CapsuleScene:
    """Random 3D capsule network with rejection sampling (spec §4/P7).

    Centers uniform in-xy with a border margin; z-centers uniform in
    `z_center_range_um` (µm around the crop's mid-plane); tilt out of the
    xy-plane uniform in [0, max_tilt_deg]. Segments closer than
    `min_sep_um` in 3D are rejected.
    """
    rng = np.random.default_rng(seed)
    px = spec.pixel_size_um
    m, n, nz = spec.fov_px_y, spec.fov_px_x, spec.n_planes
    mid_plane = (nz - 1) / 2.0
    margin_px = 2.0 / px                      # 2 µm border margin
    max_tilt = np.deg2rad(max_tilt_deg)
    segs: list[np.ndarray] = []

    def seg_endpoints_3d(cx: float, cy: float, cz_um: float, length_um: float,
                         ang: float, tilt: float) -> np.ndarray:
        # in-plane direction (cos, sin) rotated out of plane by tilt;
        # cz_um is an offset from the MID plane (µm), converted to planes
        cz_plane = mid_plane + cz_um / spec.z_step_um
        dirx = np.cos(ang) * np.cos(tilt)
        diry = np.sin(ang) * np.cos(tilt)
        dirz = np.sin(tilt)
        half = 0.5 * length_um
        return np.array([
            (cx - half * dirx) / px, (cy - half * diry) / px, cz_plane - half * dirz / spec.z_step_um,
            (cx + half * dirx) / px, (cy + half * diry) / px, cz_plane + half * dirz / spec.z_step_um,
        ])

    def seg_seg_dist_um(a: np.ndarray, b: np.ndarray) -> float:
        # 3D point-segment distances between the two segments (coarse but
        # sufficient for rejection): sample each segment at 9 points
        pts_a = np.array([a[:3] + t * (a[3:] - a[:3]) for t in np.linspace(0, 1, 9)])
        pts_b = np.array([b[:3] + t * (b[3:] - b[:3]) for t in np.linspace(0, 1, 9)])
        pts_a_um = pts_a * [px, px, spec.z_step_um]
        pts_b_um = pts_b * [px, px, spec.z_step_um]
        d = np.linalg.norm(pts_a_um[:, None, :] - pts_b_um[None, :, :], axis=-1)
        return float(d.min())

    attempts = 0
    while len(segs) < n_segments and attempts < 20000:
        attempts += 1
        length = rng.uniform(*length_range_um)
        ang = rng.uniform(0.0, np.pi)
        tilt = rng.uniform(0.0, max_tilt) if max_tilt > 0 else 0.0
        cx = rng.uniform(margin_px * px, (n - margin_px) * px)
        cy = rng.uniform(margin_px * px, (m - margin_px) * px)
        cz = rng.uniform(*z_center_range_um)
        seg = seg_endpoints_3d(cx, cy, cz, length, ang, tilt)
        # z endpoints must stay inside the plane range
        if seg[2] < 0 or seg[5] > nz - 1 or seg[5] < 0 or seg[2] > nz - 1:
            continue
        ok = all(seg_seg_dist_um(seg, s) >= min_sep_um for s in segs)
        if not ok:
            continue
        segs.append(seg)

    if len(segs) < n_segments:
        raise RuntimeError(
            f"capsule scene: only {len(segs)}/{n_segments} segments accepted "
            f"(seed={seed})")

    segs_arr = np.asarray(segs, dtype=float)
    angles = np.mod(np.arctan2(segs_arr[:, 4] - segs_arr[:, 1],
                               segs_arr[:, 3] - segs_arr[:, 0]), np.pi)
    dz = (segs_arr[:, 5] - segs_arr[:, 2]) * spec.z_step_um
    dxy = np.hypot(segs_arr[:, 3] - segs_arr[:, 0], segs_arr[:, 4] - segs_arr[:, 1]) * px
    tilts = np.degrees(np.arctan2(dz, dxy))
    scene = CapsuleScene(spec=spec, segments=segs_arr,
                         angles_rad=angles, tilts_deg=tilts)
    scene.objects = [render_capsule_unit(*seg, spec) for seg in segs_arr]
    return scene
