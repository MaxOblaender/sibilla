"""Tests for the O1 bank policies (plans/fit-speedups.md §4).

`blurred_bank_columns` builds the per-(rod × z-candidate) blurred columns:

- ``full_box`` — legacy behavior, bit-identical to the pre-O1 loop
  (full-box ``cyclic3d`` per rod × candidate).
- ``cluster``  — rods grouped into clusters whose ROI/support regions come
  within kernel reach; each cluster is convolved on its own square sub-box
  with the full-box PSF array *cyclic-cropped* (never rebuilt — the PSF build
  is grid-size dependent, measured ~1e-3 plateau, 2026-11-16). Cross-cluster
  far-tail entries are ZERO (a model decision, gated at fit level — see plan
  §4.3). Within a cluster the columns must match the legacy path to ~1e-15
  (measured), including for close rod pairs with overlapping ROIs.
"""

from __future__ import annotations

import numpy as np
from scipy import fft as sp_fft

from sibilla.defaults import Z_CANDIDATE_GRID_UM
from sibilla.fit import FitInput3D, _build_roi_masks, blurred_bank_columns, fit_3d
from sibilla.forward import cyclic3d, psf3d_cyclic
from sibilla.psf import PupilConfig
from sibilla.scene import CapsuleSpec, render_capsule_unit

PIX = 0.1135
ZCAND_INT = [int(round(z / 0.5)) for z in Z_CANDIDATE_GRID_UM]
COEFFS = {6: 0.10}


def _config(fov: int) -> PupilConfig:
    return PupilConfig(pixel_size=PIX, lambda_um=0.505, na=0.95, n_im=1.33,
                       grid_size=fov)


def _spec(fov: int, nz: int = 15) -> CapsuleSpec:
    return CapsuleSpec(fov_px_y=fov, fov_px_x=fov, n_planes=nz,
                       pixel_size_um=PIX, z_step_um=0.5, diameter_um=0.65,
                       aa_width_px=1.0)


# (x0, y0, z0, x1, y1, z1) plane units; r0/r1 are a close pair whose ROIs
# overlap (6 px apart), r2/r3 spread across the field.
RODS = np.array([
    [60, 70, 7, 78, 70, 7],
    [52, 76, 7, 72, 76, 7],
    [110, 40, 7, 126, 48, 7],
    [30, 120, 7, 44, 128, 7],
], dtype=float)


def _scene(fov: int):
    spec = _spec(fov)
    objs = [render_capsule_unit(*seg, spec) for seg in RODS]
    masks = _build_roi_masks(RODS[:, [0, 1, 3, 4]], (fov, fov), 1.5, PIX)
    idx_per_rod = [np.unique(np.where(m)[0] * fov + np.where(m)[1])
                   for m in masks]
    idx = np.unique(np.concatenate(idx_per_rod))
    return objs, idx, idx_per_rod


def _reference_column(obj, psf, dz, idx, fov):
    oc = obj if dz == 0 else np.roll(obj, dz, axis=0)
    return cyclic3d(oc, psf).reshape(psf.shape[0], -1)[:, idx]


class TestBankPolicies:
    def test_full_box_policy_is_bitwise_legacy(self):
        """policy='full_box' must reproduce the pre-O1 loop exactly."""
        fov, nz = 128, 15
        psf = psf3d_cyclic(_config(fov), COEFFS, nz, 0.5, model="confocal")
        objs, idx, _ = _scene(fov)
        bank = blurred_bank_columns(objs, psf, ZCAND_INT, idx,
                                    policy="full_box")
        for s, obj in enumerate(objs):
            for i, dz in enumerate(ZCAND_INT):
                ref = _reference_column(obj, psf, dz, idx, fov)
                assert np.array_equal(bank[s][i], ref)

    def test_cluster_near_field_matches_full_box(self):
        """Within-kernel-reach cropping is machine-exact at the rod's own ROI
        pixels — including the close pair with overlapping ROIs (plan §4.2)."""
        fov, nz = 128, 15
        psf = psf3d_cyclic(_config(fov), COEFFS, nz, 0.5, model="confocal")
        objs, idx, idx_per_rod = _scene(fov)
        bank = blurred_bank_columns(objs, psf, ZCAND_INT, idx, idx_per_rod,
                                    policy="cluster")
        for s in range(len(objs)):
            for i, dz in enumerate(ZCAND_INT):
                ref = _reference_column(objs[s], psf, dz, idx_per_rod[s], fov)
                pos = np.searchsorted(idx, idx_per_rod[s])
                got = bank[s][i][:, pos]
                scale = max(np.abs(ref).max(), 1e-30)
                assert np.abs(ref - got).max() / scale < 1e-10, (s, i)

    def test_cluster_zeroes_cross_group_tail(self):
        """Two rods far apart (> reach + ROI/support radii) form separate
        clusters; cross-cluster entries must be ZERO (the documented model
        decision), while each rod's own ROI stays exact. The legacy bank has
        non-zero entries there — for this geometry they are cyclic-wrap
        artifacts (separation > half-box), plan §4.3."""
        fov, nz = 256, 15
        psf = psf3d_cyclic(_config(fov), COEFFS, nz, 0.5, model="confocal")
        spec = _spec(fov)
        skel = np.array([[60, 60, 7, 75, 60, 7],
                         [200, 200, 7, 215, 205, 7]], dtype=float)
        objs = [render_capsule_unit(*s, spec) for s in skel]
        masks = _build_roi_masks(skel[:, [0, 1, 3, 4]], (fov, fov), 1.5, PIX)
        idx_per_rod = [np.unique(np.where(m)[0] * fov + np.where(m)[1])
                       for m in masks]
        idx = np.unique(np.concatenate(idx_per_rod))
        bank = blurred_bank_columns(objs, psf, ZCAND_INT, idx, idx_per_rod,
                                    policy="cluster", kernel_reach_px=20)
        legacy = blurred_bank_columns(objs, psf, ZCAND_INT, idx,
                                      policy="full_box")
        for s in range(2):
            pos = np.searchsorted(idx, idx_per_rod[s])
            for i in range(len(ZCAND_INT)):
                ref = legacy[s][i][:, pos]
                got = bank[s][i][:, pos]
                scale = max(np.abs(ref).max(), 1e-30)
                assert np.abs(ref - got).max() / scale < 1e-10, (s, i)
        # the other rod's ROI pixels: legacy carries (wrap) tail entries,
        # cluster zeroes them
        for s, t in ((0, 1), (1, 0)):
            pos_t = np.searchsorted(idx, idx_per_rod[t])
            assert np.abs(bank[s][0][:, pos_t]).max() == 0.0, s
            assert np.abs(legacy[s][0][:, pos_t]).max() > 0.0, s


class TestPhaseRampZCandidates:
    """O2 (plans/fit-speedups.md): per-candidate np.roll along z == multiplying
    the object's FFT by exp(-2*pi*i*kz*dz/nz) (exact identity, different
    rounding). Saves the per-candidate object FFT: 1 forward FFT per rod
    instead of n_candidates. Applies ONLY to the cluster sub-box path —
    full_box and the single-cluster fallback stay roll-based (legacy anchors).
    """

    def test_phase_ramp_matches_roll_all_candidates(self):
        fov, nz = 128, 15
        psf = psf3d_cyclic(_config(fov), COEFFS, nz, 0.5, model="confocal")
        objs, idx, idx_per_rod = _scene(fov)
        # explicit candidate list: negatives, zero, non-grid magnitudes
        zc = [-3, -2, -1, 0, 1, 2, 4]
        roll = blurred_bank_columns(objs, psf, zc, idx, idx_per_rod,
                                    policy="cluster", z_phase_ramp=False)
        ramp = blurred_bank_columns(objs, psf, zc, idx, idx_per_rod,
                                    policy="cluster", z_phase_ramp=True)
        for s in range(len(objs)):
            for i in range(len(zc)):
                a, b = roll[s][i], ramp[s][i]
                scale = max(np.abs(a).max(), 1e-30)
                assert np.abs(a - b).max() / scale < 1e-9, (s, zc[i])

    def test_phase_ramp_default_off(self):
        """The flag defaults to False (legacy roll path) until gated."""
        import inspect
        sig = inspect.signature(blurred_bank_columns)
        assert sig.parameters["z_phase_ramp"].default is False


class TestFit3DBankPolicy:
    def test_cluster_policy_agrees_with_full_box(self):
        """End-to-end: same data, both policies, rods within one cluster ->
        same fitted coefficients (the bank differs only at ~1e-15)."""
        fov, nz = 96, 15
        cfg = _config(fov)
        psf = psf3d_cyclic(cfg, COEFFS, nz, 0.5, model="confocal")
        spec = _spec(fov)
        skel = np.array([[44, 48, 7, 60, 52, 7],
                         [32, 60, 7, 46, 66, 7]], dtype=float)
        objs = [render_capsule_unit(*s, spec) for s in skel]
        rng = np.random.default_rng(7)
        data = np.full((nz, fov, fov), 250.0)
        for obj in objs:
            data += 4500.0 * cyclic3d(obj, psf) / obj.max()
        data = rng.poisson(data).astype(np.float64)
        z_planes = (np.arange(nz) - (nz - 1) / 2) * 0.5

        results = {}
        for policy in ("full_box", "cluster"):
            fit_in = FitInput3D(
                data=data, config=cfg, skeleton=skel, z_planes_um=z_planes,
                nominal_budget=4500.0, fluorescent_diameter_um=0.65,
                z_candidate_um=Z_CANDIDATE_GRID_UM, psf_model="confocal",
                n_multi_start=1, max_nfev=30, n_outer_sweeps=2,
                bank_policy=policy,
            )
            results[policy] = fit_3d(fit_in)
        a, b = results["full_box"], results["cluster"]
        for mode in (5, 6, 7, 8):
            assert abs(a.coeffs.get(mode, 0.0) - b.coeffs.get(mode, 0.0)) < 1e-9
        assert np.allclose(a.segment_z_um, b.segment_z_um)

    def test_fit3d_phase_ramp_agrees_with_roll(self):
        """End-to-end O2 gate: cluster path with the phase-ramp z-candidates
        must give the same fit as the roll-based cluster path (default)."""
        fov, nz = 96, 15
        cfg = _config(fov)
        psf = psf3d_cyclic(cfg, COEFFS, nz, 0.5, model="confocal")
        spec = _spec(fov)
        skel = np.array([[44, 48, 7, 60, 52, 7],
                         [32, 60, 7, 46, 66, 7]], dtype=float)
        objs = [render_capsule_unit(*s, spec) for s in skel]
        rng = np.random.default_rng(7)
        data = np.full((nz, fov, fov), 250.0)
        for obj in objs:
            data += 4500.0 * cyclic3d(obj, psf) / obj.max()
        data = rng.poisson(data).astype(np.float64)
        z_planes = (np.arange(nz) - (nz - 1) / 2) * 0.5

        results = {}
        for ramp in (False, True):
            fit_in = FitInput3D(
                data=data, config=cfg, skeleton=skel, z_planes_um=z_planes,
                nominal_budget=4500.0, fluorescent_diameter_um=0.65,
                z_candidate_um=Z_CANDIDATE_GRID_UM, psf_model="confocal",
                n_multi_start=1, max_nfev=30, n_outer_sweeps=2,
                bank_policy="cluster", z_phase_ramp=ramp,
            )
            results[ramp] = fit_3d(fit_in)
        a, b = results[False], results[True]
        for mode in (5, 6, 7, 8):
            assert abs(a.coeffs[mode] - b.coeffs[mode]) < 1e-9, mode
        assert np.allclose(a.segment_z_um, b.segment_z_um)
