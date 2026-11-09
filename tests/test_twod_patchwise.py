import numpy as np
from scipy import fft as sp_fft
from sibilla.twod import patchwise
from sibilla.twod.psf2d import psf2d

def test_patch_grid_covers_whole_image():
    tiles = patchwise.patch_grid((640, 700), size=256, overlap=0.25)
    ys = [t[1] for t in tiles] + [t[1] + 256 for t in tiles]
    xs = [t[2] for t in tiles] + [t[2] + 256 for t in tiles]
    assert min(ys) == 0 and max(ys) >= 640
    assert min(xs) == 0 and max(xs) >= 700

def test_patch_grid_last_patch_ends_at_edge():
    tiles = patchwise.patch_grid((512, 512), size=256, overlap=0.25)
    assert (0, 0) in [(t[1], t[2]) for t in tiles]
    assert max(t[1] + 256 for t in tiles) == 512

def test_feather_weights_peak_in_centre_and_zero_at_edges():
    w = patchwise.feather_weights(64, overlap=0.25)
    assert w.shape == (64, 64)
    assert w[0, :].max() < 0.2
    assert w[32, 32] > 0.5
    assert np.all(w >= 0) and w.max() <= 1.0

def test_recovery_metric_improves_after_correct_deblur():
    rng = np.random.default_rng(0)
    obj = np.clip(rng.random((256, 256)), 0.02, 0.8).astype(np.float32)
    blurred = np.real(sp_fft.ifft2(sp_fft.fft2(obj) * sp_fft.fft2(psf2d(10.0, "CH1", 256))))
    assert patchwise.recovery_metric(blurred, obj) > 1.2

def _tissue_tile(seed=0, n=512):
    """Mostly-empty tile with one smooth correlated structure spanning the
    patch grid — the physical preconditions the pipeline assumes: (a) an
    empty background so the noise floor measures sensor-like noise,
    (b) shared low-frequency structure across patches so per-patch L2
    gains are O(1). (White-noise tiles violate both: measured Task 6.)"""
    rng = np.random.default_rng(seed)
    tile = np.full((n, n), 0.02, np.float32)
    tile += rng.normal(scale=0.005, size=tile.shape).astype(np.float32)
    F = sp_fft.fft2(rng.normal(size=(n, n)))
    fy = sp_fft.fftfreq(n)[:, None]
    fx = sp_fft.fftfreq(n)[None, :]
    F *= np.exp(-((fx**2 + fy**2) / (2 * 0.08**2)))
    tex = np.real(sp_fft.ifft2(F))
    tex = (tex - tex.min()) / (tex.max() - tex.min())
    yy, xx = np.mgrid[0:n, 0:n]
    region = np.hypot(yy - n / 2, xx - n / 2) < n * 0.35
    tile[region] += 0.6 * tex[region]
    return np.clip(tile, 0.0, None).astype(np.float32)

def test_patchwise_returns_one_record_per_patch():
    rng = np.random.default_rng(0)
    obj = np.clip(rng.random((512, 512)), 0.02, 0.8).astype(np.float32)
    out, recs = patchwise.deconvolve_patchwise(obj, "CH1", size=256, n_iter=2)
    assert out.shape == obj.shape
    assert len(recs) == len(patchwise.patch_grid(obj.shape, 256, 0.25))
    assert all(r.size == 256 for r in recs)

def test_background_only_arm_same_geometry_different_pixels():
    obj = _tissue_tile(2)
    blurred = np.real(sp_fft.ifft2(sp_fft.fft2(obj) * sp_fft.fft2(psf2d(10.0, "CH1", 512)))).astype(np.float32)
    out_full, recs_full = patchwise.deconvolve_patchwise(blurred, "CH1", size=256, n_iter=2)
    out_bg, recs_bg = patchwise.deconvolve_patchwise(blurred, "CH1", size=256, run_rl=False)
    assert len(recs_bg) == len(recs_full)
    assert all(r.iterations == 0 for r in recs_bg)
    assert not np.allclose(out_bg, out_full)

def test_reassembly_no_ringing_at_patch_border():
    obj = _tissue_tile(1)
    out, _ = patchwise.deconvolve_patchwise(obj, "CH1", size=256, n_iter=2)
    col = out[:, 256].mean()
    neigh = np.mean([out[:, 250].mean(), out[:, 262].mean()])
    assert abs(col - neigh) / (neigh + 1e-6) < 0.15

def test_rl_early_stops_on_tissue_fixture():
    """Spec §7 early stop (round-3 pin): with the §7 metric wired into
    deconvolve_patchwise, RL must not run all 30 iterations on the tissue
    fixture — the band-limited spectral distance to the reference plateaus
    (convergence) or grows (divergence), and either way the plateau
    detector stops the run; PatchRecord.iterations reports it."""
    obj = _tissue_tile(2)
    blurred = np.real(sp_fft.ifft2(sp_fft.fft2(obj) * sp_fft.fft2(psf2d(10.0, "CH1", 512)))).astype(np.float32)
    out, recs = patchwise.deconvolve_patchwise(blurred, "CH1", size=256, n_iter=30)
    full = [r for r in recs if r.iterations > 0]
    assert full, "expected at least one patch to run RL"
    assert all(r.iterations < 30 for r in full)

def test_indeterminate_patches_skip_rl_like_unrecoverable():
    """Round-4 pin: with ``min_dz_identification > 0``, patches whose dz
    fit cannot exclude dz = 0 (here every patch of the in-focus tissue
    tile — measured dz_err = inf: the MTF² model is stationary in dz at
    the 0 bound, so the Jacobian is singular) take the skip-RL path with
    their own record class — dz_um=nan, iterations=0, recovery=None,
    unrecoverable=False, indeterminate=True — and the reassembled tile is
    pixel-identical to the background-only arm (L1+L2+feather only). The
    control arm never sets the flag (dz is not attempted there)."""
    obj = _tissue_tile(2)
    out, recs = patchwise.deconvolve_patchwise(
        obj, "CH1", size=256, n_iter=2, min_dz_identification=50.0)
    assert recs
    assert all(r.indeterminate and not r.unrecoverable for r in recs)
    assert all(np.isnan(r.dz_um) and r.iterations == 0 and r.recovery is None
               for r in recs)
    out_bg, recs_bg = patchwise.deconvolve_patchwise(
        obj, "CH1", size=256, run_rl=False, min_dz_identification=50.0)
    assert all(not r.indeterminate for r in recs_bg)
    assert np.array_equal(out, out_bg)
