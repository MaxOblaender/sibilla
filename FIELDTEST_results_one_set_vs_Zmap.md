# Field-test results — what the A/B/C fits tell us (2026-11-03)

Companion to `FIELDTEST_one_set_vs_Zmap.md` (protocol, §6 results now filled by
this doc). Status: **2026-11-05 UPDATE: the corrected 8-fit diagnostic is
complete — ALL 8 pin at ±1.5 with correct z and clean rods, and the C3
z0-vs-z3 probe pins in BOTH arms. Verdict per the §8 matrix: the per-rod
z-shift is NOT the escape channel — Z5–Z8 ride a flat cost valley and must
be reported as bounds / a regularised posterior, not point estimates. §9
adds the cause analysis: the pinning driver is diffuse out-of-focus haze
from dense regions (per-plane-constant background can't model it), NOT
neighbour tube overlap, orientation, rod quality, or rod count — which is
why the sparse, dim early crop (4 rods, tiny lever) read out sensible
small coefficients while the bright, crowded A/B/C regions slide to the
bound.**
**§10 (2026-11-05, later): the user-drawn sparse/deep campaign — shallow
tiles (T, 3–7 µm) de-pin at ~0.1–0.2 w and reproduce the October early crop;
the mid-depth tile (S, 40–56 µm) pins even though sparse; the
bright+isolated+deep cell does not exist in this specimen; and the Z5/Z6
axis is rod-composition-driven (LOO) — report amplitudes, never signed
axes.**
Everything below §8 is the original analysis — kept for the record, but its
main conclusion (large aberration) is retracted.

Fitter: server copy `zernike_test` (`fit.py` md5
`4fcb3e0c8b910f6677a603b054b3905e`, **same commit as this branch** for the
original field test; `89df1222…` = commit `50f2aed` for the diagnostic),
`fit_3d`, confocal model, `n_multi_start=2`, `max_nfev=150`.

---

## 8. STATUS UPDATE 2026-11-03 (post-launch): the z-derivation bug

While the first diagnostic run was in flight, a per-rod verification
(axial intensity profile at each derived capsule vs. the stored z) revealed
a **one-line bug in the throwaway field-test runner's `derive()`**:

```python
za = find_z(sA, ax, ay, zc - zlo)   # BUG: z_guess must be the window-center
                                    # plane in ABSOLUTE crop coords
```

`find_z` returns `z_guess + (sharpest_offset_from_window_center)`. The 7-
plane window `sA` starts at crop plane `zlo` (center = `zlo + 3`). Passing
`z_guess = zc − zlo` (= 3 for interior rods) makes the stored z a
*window-local* value, off from the true crop plane by `(zc − 3)` — i.e.
**every interior rod was placed 2–9 µm ABOVE its real depth**.

Verified rod-by-rod on all three regions: for every interior rod
`stored_cz = true_cz − (zc − 3)` holds exactly, and `true_cz` (recomputed)
matches the full-crop axial-profile argmax within ±2 planes for 23/27 rods.

### Impact — retracted / revised conclusions

- **All 15 field-test fits, and the first diagnostic launch, are
  confounded.** Each rod carried up to ~9 µm of unmodeled defocus (≈ 6× the
  fitter's ±1.5 µm z-search, ≈ 3× the diagnostic's ±3 µm). Per
  `SUPERVISOR_one_set_vs_Zmap.md` §6, unmodeled defocus leaks into Z5–Z8 —
  and a *uniform* bound-pinning with seed-flipping signs is exactly what a
  systematic z bias produces.
- **§3.1's "the aberration is ≥ 0.5 waves" is RETRACTED.** It was measured
  on fits with systematically wrong rod depths. Current knowledge: the
  aberration magnitude is *unmeasured*.
- **C's "stable Z7/Z8 ≈ 0" structure is suspect** — possibly an artifact of
  the z bias.
- **The weekend region fits are NOT affected.** They ran via
  `server_fit.py`, whose `find_z` is called with the window centered at
  `zc` and `z_guess = zc` (correct). **region3's pinning at ±0.5 therefore
  remains a real, unexplained problem** (now with correct z as a further
  check: it pins even though its z was right).
- **The first diagnostic result (C_full_0, pinned at ±1.5 with z-residuals
  saturating the new ±3 µm edge) is no longer clean evidence** that the
  bound/z-window are still too narrow — the z bug alone can produce it.

### Rod-selection check (the question that triggered this)

Per-rod review (brightness, lateral peak offset, neighbor clumps, length,
axial peak; figures on the server in `diag_review/`):

- **The drawn lines are mostly fine** — lateral offsets ≤ ~10 px (mask is
  ~13 px wide), lengths match, nothing saturated.
- **4 of 27 rods are bad** even after the z fix: **A r4, A r6, B r5, B r8**
  — the drawn line sits on a shallower dimmer rod while a brighter peak is
  6–18 planes away (line on the wrong rod, or rod too dim: A r4 SNR 2.0).
- Region C's rods are all good after the fix.

### Corrected run (in progress, nodeasb01g)

- First diagnostic launch (12:44 UTC, buggy z + wide bound) **aborted at
  ~2.3 h** after the bug was found; only `C_full_0` completed. Preserved
  (NOT deleted) in `weekend_logs/fieldtest_diag_buggy_z/` with
  `PROVENANCE.txt`.
- **Corrected relaunch, 15:06→15:10 UTC**, wide limits kept
  (±1.5 waves, 13 z-candidates) — 8 fits:
  - A, B: `full` × seeds 0,1 (**all 9 rods** — directly comparable to the
    original field test, only z corrected) **plus** `full_clean` seed 0
    (7 rods, the 4 bad-z rods excluded);
  - C: `full` × seeds 0,1 (no exclusions). C has run since 15:06 (its
    job list is identical in both runner versions).
  - Resumable; watcher writes `DONE`/`ABORTED` + `watcher.log`; original
    `weekend_logs/fieldtest/` untouched.
  - One mishap: the first fixed-z launch (15:06) had a rod-set bug in its
    own job design (A/B `full` jobs ran with 7 instead of 9 rods); killed
    at 15:09 before any JSON was written (partial logs preserved in the
    archive dir), A/B restarted at 15:10 on the corrected runner
    (md5 `8a4806dc`).
- **The design now isolates z:** buggy-z + wide bound (archived C_full_0)
  vs corrected-z + wide bound (running). If the corrected fits come off the
  bound, the pinning was the z bug; if they still pin at ±1.5 with correct
  z *and* clean rods, the bound/model suspicion is revived (and region3,
  which had correct z, is the reference case).

### Progress snapshot (2026-11-04 14:20 UTC) — 5/8 fits done, all still pinned

C is complete; **B full s1 finished 07:54 UTC and is bit-identical to
s0** (same cost to the integer, same z-residuals — under a shared-
coefficient fit with reset z-state this is a fixed point, not evidence
of seed stability, see the caveat below). A is on its second full seed,
B is on `full_clean`:

| job | Z5, Z6, Z7, Z8 (waves) | pinned / pinned05 | z-residuals (µm) |
|---|---|---|---|
| C full s0 (175 min) | +1.5, +1.5, +0.011, +0.076 | yes / yes | ±3.0 on 2/9 rods (r8, r9) |
| C full s1 (188 min) | +1.5, +1.5, +0.011, +0.076 | yes / yes | identical to s0 (exact) |
| A full s0 (568 min) | −1.5, −1.5, +0.821, −0.140 | yes / yes | ±3.0 on 3/9 rods; bad-z r6 at 2.5 |
| B full s0 (548 min) | −0.039, +1.5, +1.5, −0.389 | yes / yes | ±3.0 on 1/9 rods (bad-z r5); r7 at 2.5 |
| B full s1 (457 min) | −0.039, +1.5, +1.5, −0.389 | yes / yes | **identical to s0 (bit-exact)** |
| A full s1, A full_clean, B full_clean | — running — | | |

Interim read (per the decision tree below): C's **seeds agree exactly**
(seed stability is good), but the fits do **not** come off the ±1.5 bound
with corrected z — and A/B's full_0 still carry the 2 bad-z rods each
(the full_clean jobs are the contamination check). Since region3 (correct
z, clean rods) pinned at ±0.5 too, the "bound/model suspect" branch stays
alive. The z-residuals no longer saturate uniformly (3 of 9 rods are at
the ±3 µm edge, not all 9), which is consistent with the z bug being gone;
what remains is the bound/model question.

### Decisive probe, run 2026-11-04: the "1.5 waves" explains almost nothing (c-sweep)

While the 8-fit run was in flight, a forward-model probe (`resid_probe`,
server `weekend_logs/resid_probe/`, local copy gitignored) was run:
the fitter's **own** model (exact capsules, zero z-search) evaluated at
several coefficient vectors, measuring the fraction of the tube-union
signal it explains (per-rod amplitudes + per-plane background free —
the same degrees of freedom the fit has, minus z).

| coeff (Z5,Z6,Z7,Z8) | magnitude | A explained | C explained |
|---|---|---|---|
| 0,0,0,0 (no aberration) | 0 w | 55.5 % | **87.2 %** |
| region's pinned result | 2.1–2.5 w | 59.6 % | 91.4 % |
| 2× pinned | 4.2–4.3 w | 59.8 % | 91.9 % |

**Reading:** setting the coefficients to the *pinned* values improves the
fit by only **+4.1 points** over no aberration at all; doubling to 3
waves buys **+0.2–0.6 more**. If the data really contained ~2 waves of
Z5–Z8, matching them would collapse the residual — they don't. The data
are **~diffraction-limited** (consistent with the images looking good),
and the pinning is an optimizer sliding down a **flat cost valley** to
the bound, not the data demanding large aberration. Consequences:

- The ±1.5 bound is **not** the limiting factor (widening won't help).
- The free per-rod **z-shift is the prime suspect** for absorbing what
  should be aberration → the running z-sweep (A/C × z3/z0, C3-3rod) is
  the decisive test; design for the z-fixed outcome is in
  `plans/z-range-derivation.md`.
- Seed "stability" from this run is weak evidence: `fit_3d` resets all
  z-state to the midpoint per start, so both seeds walk the same path
  (bit-identical B/C results confirm it's a fixed point, not stability).
- Also probed (same dir): per-rod **zero-aberration residual** — a
  perfect capsule model leaves 35 % (C) / 67 % (A,B) of tube signal
  unexplained; `resid_{A,B,C}.tif` are scrollable 3D maps of it.
  Figures: `csweep_figure.png`.

**Width probe (2026-11-04, `width_probe.json`): rod width is not the
cause.** The model assumes a fixed capsule diameter (0.65 µm); asked
whether the rods' actual width variation matters, the same forward
model was evaluated at W = 0.5–0.8 µm.

*How it was tested.* For each width, the fitter's *own* code re-drew
every rod at that thickness, blurred it with the real PSF, and matched
it to the **real measured pixels** inside each rod's 1.5 µm tube
(brightness and background free — the same freedom the real fit has).
The score is "what fraction of the real light is left unexplained."
Only the width changes between runs (box, positions, z, aberration are
identical), so any score difference is caused by the width alone. The
setup was verified first: at W = 0.65 µm it reproduced the independent
residual-probe reference (C: 0.357 vs 0.353), i.e. it ran on
correctly-registered data.

*What it does NOT measure.* Because the tube is ~2× wider than the rod,
the rod fits inside it at every width — so this test says "does the
assumed width change the fit quality / drive the coefficients," **not**
"what is each rod's true width." For that, measure a perpendicular
profile through an in-focus rod and deconvolve the PSF out
(object ⊛ PSF → solve for object width), or use the synthetic
known-width bias test (2026-10-29, pre-flight P3). Not on the critical
path since the width knob barely moves the cost.

| region | unexplained @ c=0, W=0.5 | W=0.65 | W=0.8 | total width effect |
|---|---|---|---|---|
| A | 0.6712 | 0.6670 | 0.6632 | 0.8 pts |
| B | 0.6925 | 0.6921 | 0.6915 | 0.1 pts |
| C | 0.3605 | 0.3572 | 0.3542 | 0.6 pts |

The trend is monotone toward *wider* (true rods are likely a touch
thicker than 0.65 µm) but the total effect is **≤0.8 points** — an
order of magnitude smaller than the aberration's own effect (3–6 pts)
and far too small to drive the pinning. Brightness differences are
absorbed by the free per-rod amplitude. Rod width/brightness variation
is **ruled out** as a pinning cause; the 0.65 µm default is fine for
now (a 0.7 µm default would buy ~0.2 pts — not worth a change).

### FINAL read-out (2026-11-05): 8/8 done — all pin at ±1.5, clean rods included

Watcher `DONE` 01:42 UTC 2026-11-05; all 8 JSONs in
`weekend_logs/fieldtest_diag/` (fit.py md5 `89df1222`, commit `50f2aed`).
The three jobs that were pending at the snapshot above:

| job | Z5, Z6, Z7, Z8 (waves) | pinned / pinned05 | z-residuals (µm) |
|---|---|---|---|
| A full s1 | −1.5, −1.5, +0.821, −0.140 | yes / yes | **identical to s0** (coeffs + z-residuals) |
| A full_clean s0 (7 rods) | −1.5, −1.5, +0.843, −0.149 | yes / yes | ±3.0 on 3/7 rods (r1, r5, r7) |
| B full_clean s0 (7 rods) | +1.5, +1.5, +1.5, +1.186 | yes / yes | r4 at 3.0, r7 at 2.5 |

- **All 8 fits pin at ±1.5** (corrected z, wide bound). Seeds are
  bit-identical in ALL three regions now — the documented fixed point
  (z-state reset per start), not stability.
- **The de-contamination check does not de-pin.** Removing the bad-z
  rods (A r4/r6, B r5/r8) leaves A nearly unchanged (Z7 0.821 → 0.843);
  B ends with three modes AT the bound and Z5's sign flipped
  (−0.039 → +1.5) — flat-valley sign instability, not a contamination
  response.
- Decision-tree outcome: "still pinned at ±1.5 with correct z *and*
  clean rods", now confirmed against the c-sweep (below) and region3's
  ±0.5 reference. **The ε/σ field comparison — the original field-test
  question — is not answerable from Z5–Z8 point estimates at this SNR.**

### C3 z0-vs-z3 probe (DONE 22:09 UTC 2026-11-04): both arms pinned

`c3probe/` — 3 clean C rods (r2, r8, r9), {z fixed, ±3 µm} × {bound 1.5, 5}:

| job | Z5, Z6, Z7, Z8 (waves) | pinned | cost | z-res (µm) |
|---|---|---|---|---|
| C3 z3_b15 | +1.500, +1.500, +0.210, +0.893 | yes | 479 577 | 0, −1, −3 |
| C3 z0_b15 | +1.500, +1.500, +0.225, +0.906 | yes | 483 897 | z fixed |
| C3 z3_b5 | −4.073, −1.236, −0.002, +0.078 | no | 406 211 | 0, +2.5, −2.5 |
| C3 z0_b5 | −3.653, −1.648, +0.003, +0.093 | no | 417 310 | z fixed |

- **Primary contrast at bound 1.5: both arms pin**, with nearly
  identical coefficients; freeing z buys 0.9 % cost. Matrix verdict:
  **"both pinned" — the per-rod z-shift is NOT the escape channel; the
  Z5–Z8 flat valley is intrinsic.** Next per the matrix: report upper
  bounds / regularised posterior; coefficient prior (O-z4);
  more/better-oriented rods; `plans/tiled-field-scan.md`.
- Caveats that keep the 9-rod replication worthwhile: C3 has weak
  orientation spread (72°/95°/178°), and r2's fitted amplitude is 0 in
  all four runs (its tube signal is absorbed by the free per-plane
  background) — effectively a 2-rod probe. Hence the relaunched 9-rod
  z_sweep (below).
- **Bound-±5 slide (interpretation, not established):** loosening the
  bound to 5 drops the cost ~14–15 % toward Z5 ≈ −4 / Z6 ≈ −1.6
  (Z7/Z8 ≈ 0), but the free per-rod amplitudes jump ~4–8× (r8
  0.08 → 0.29–0.34; r9 0.02 → 0.16–0.20) — the amplitude↔aberration
  degeneracy (bigger aberration attenuates the PSF; free amplitude
  compensates). Lower cost ≠ more explained signal; the discriminating
  metric `unexpl_frac_at_fit` is not in these JSONs (the c3probe runner
  doesn't compute it) — the relaunched z_sweep records it.

### The 9-rod z_sweep was lost to a metric bug; fixed and relaunched 2026-11-05

All four 9-rod jobs (A z3_b15 serial; A z0 / C z0 / C z3 one-shots)
**completed `fit_3d`, then crashed in `run_z_sweep.py`'s `unexpl()`
BEFORE the JSON dump**: the per-plane background block was built as
`np.repeat(np.eye(nzp), len(idx), axis=1)` → (nzp, nzp·L) instead of
(nzp·L, nzp), so `hstack` raised the 326952/345096-vs-36 ValueError and
`res` died with the process — **≈35 h of completed fits discarded,
nothing written** (A z3_b15 ≈15 h; A z0 ≈6.5 h; C z0 ≈4.3 h; C z3 ≈8.8 h).
Same ValueError `width_probe.py` hit on 2026-11-04 (fixed and validated
there by its server run).

Fix + verification (2026-11-05): the validated construction
(`A_bg[rows, rows // L] = 1.0`) was ported into `run_z_sweep.py`
(pre-patch copy kept as `run_z_sweep.py.buggy_unexpl_20261105`).
**Verified by unit test, not by DRY**: `test_unexpl_patch.py` runs the
patched function on the exact crash shapes (nzp=36) — it returns a valid
fraction, and the per-plane background absorbs a constant offset exactly
(residual conserved to 3 ppm; the score is ‖r‖/‖b‖ so only the
normaliser moves). The runner's own DRY mode timed out at 9 min inside
`fit_3d` before ever reaching `unexpl()`, so the b15 9-rod relaunch is
**held** (nothing lost — it can be launched in one command); the
bound-0.5 wave instead carries the contrast (§9.6).

## 9. WHY THE FIELD-TEST REGIONS PIN — and why the sparse early crop worked (2026-11-05)

Follow-up diagnosis after §8: the corrected fits all pin, and the question was
*is it the z-range, rod quality, rod count, or the regions themselves?* Ruling
each out, in order. (All throwaway scripts in `/asbdata/moblaende26/zerniky/`
on nodeasb01g: `neighbour_light2.py`, `orientation.py`, `test_unexpl_patch.py`,
`diag_unexpl_bg.py`.)

### 9.1 Rod count vs outcome — the early (sparse) regions were the control

The *original* pre-field-test fits (2–7 rods each, auto-extracted from the
sparse regions) were almost all **off-bound and sensible**:

| fit | rods | coeffs (max\|Z\|) |
|---|---|---|
| region4 | 2 | +0.32, −0.08, +0.06, +0.02 (0.32) |
| region5 | 2 | +0.09, +0.20, +0.01, −0.02 (0.20) |
| region6 | 2 | +0.18, −0.15, −0.06, −0.06 (0.18) |
| tright | 3 | +0.14, −0.16, +0.05, −0.04 (0.16) |
| merged | 5 | −0.22, +0.31, +0.01, −0.05 (0.31) |
| **region3** | **7** | **−0.5, −0.5, −0.04, −0.5 (PINNED)** |
| C3 (field test) | 3 | +1.5, +1.5, +0.2, +0.9 (PINNED) |
| C / A / B (field test) | 9/9/9 | all PINNED at ±1.5 |

And the early 4-rod crop (your working example), `fit_report.json`:
Z5=+0.180, Z6=+0.123, Z7=+0.008, Z8=−0.015 — off-bound, sensible.

So "sparse → sensible, dense → pinned" is real. But rod **count alone** is not
the driver: C3 is only 3 rods and pins, while region4/5/6 are 2 rods and don't.
The distinguishing variable is **the light in each tube**.

### 9.2 Neighbour light — tube overlap is tiny; the haze is diffuse

`neighbour_light2.py`: fraction of each rod's own 1.5 µm tube covered by
neighbour rods' tubes (derived z, |Δz|≤4 planes), joined with the per-rod
zero-aberration unexplained fraction from `resid_probe`:

| region | rods with overlap>0 | worst | Pearson r(overlap, unexpl) |
|---|---|---|---|
| A | 2/9 (r3 0.75, r9 0.72) | 0.75 | −0.06 |
| B | 1/9 (r1/r2 ≤0.08) | 0.08 | +0.04 |
| C | 2/9 (r3 0.05, r7 0.07) | 0.07 | +0.18 |

**Direct tube overlap is NOT the poison** — it is ≤0.08 for most rods and
*uncorrelated* with the residual (A r8: overlap 0.00, unexplained 0.96).

The real unmodelled light is **diffuse out-of-focus haze from the dense 3-D
neighbourhood**. The model's background is *one constant per plane*; look what
it has to absorb at c=0:

| region | fitted per-plane background (csweep c=0) | explained @ c=0 |
|---|---|---|
| A | **139 → 354 counts (2.6× swing)** | 55.5 % |
| C | 126 → 147 (flat) | 87.2 % |

In A, ~a third of every tube is haze the per-plane-constant background cannot
follow — that is the irreducible residual, and it is what gives the optimizer a
downhill slope into the bound (a wider aberrated PSF buys +4 pts of haze-shape
there). C has almost no haze, so its valley is flat — but with free amplitudes
it slides anyway (C3).

**The early crop proves contamination ≠ pinning:** re-extracting its 9
segments, fitted **seg4 has 90 % of its tube covered by an *unfitted*
segment** (a 0.28 µm split artifact — pure unmodelled light in a fitted rod)
and the fit *still* returned +0.18/+0.12. The lever there was tiny (amps
0.004–0.019), so contamination set a residual floor but no slope to slide down.

### 9.3 Orientation — ruled out

`orientation.py`: rod-direction spread. All sets span 85–90° and cover 5–6 of
6 fifteen-degree sectors (A 88.8°, B 84.6°, C 90.0°, early 4-rod 76.5°).
Orientation is **adequate everywhere** — not the limiting factor.

### 9.4 "z0 can't explain anything, so Z5–Z8 must compensate?" — no

`z_candidate_um=(0,)` does **not** disable the depth model: the PSF is still
evaluated at each rod's *own* depth (defocus on plane k = `(k−z_s)·z_step`);
the candidate grid is only a correction on top of the derived z. C3 showed that
correction is worth 0.9 % of cost and ~0.01 w of coefficients. If Z5–Z8 were
compensating z-error, freeing z would *shrink* them — it doesn't (1.500 →
1.500; at bound 5 they even *grow* to −4.07, and each extra wave buys <0.6
pts — the signature of an unconstrained parameter, not compensation).

### 9.5 Can sparse-but-more rods compensate the "lever" problem? — yes, pick by SNR

The lever scales with **total modelled signal** (brightness × number of
independent rods); the hazard scales with **local haze** — independent knobs.
C3's bright rods (amps 0.08–0.34) have a 30–100× bigger lever than the early
crop (0.004–0.019). Caveat: a rod can be perfectly isolated *and* useless —
C3 r2 has 0 % contamination but a fitted amplitude of exactly 0 (its tube
light is all background). So selection should rank by (1) tube signal / peak
SNR, (2) 3-D isolation (no neighbour within ~4 planes), (3) flat surrounding
ring (the haze test), (4) orientation spread — **not by rod count or
cleanliness alone**. That is the first, defensible tile of
`plans/tiled-field-scan.md`.

### 9.6 Consequence for dense/light regions — and the fix that's actually needed

**Dense, light regions are bad for fitting, for two compounding reasons:**
(i) dense → out-of-focus haze that the per-plane-constant background cannot
model (drives A to 55 %); (ii) light + isolated → big lever, so the fitter has
room to slide down the haze-slope to the bound. Neither is a bug; it's the
model. To fit dense regions you need a **spatially-varying (ring/haze)
background** in the forward model, *not* better Zernikes. To get a real
aberration estimate on the present data, fit **sparse, bright, isolated**
rods at bound 0.5.

**In flight (2026-11-05):** the bound-0.5 family — `C_z3_b05_c3` (3 bright
C rods, the live end-to-end proof of the `unexpl()` fix), `A_z3_b05_fullclean`
(7), `B_z3_b05_fullclean` (7), `C_z3_b05_full` (9) — 4 parallel nohup jobs
started 08:09 UTC, watcher → `weekend_logs/z_sweep/DONE.b05`. This tests
"bright+isolated vs dim+crowded at the bound that matters (0.5)".

### Runs inventory

**Superseded — the current two-node inventory lives in §10.10.** All
throwaway runners live in `/asbdata/moblaende26/zerniky/` on the shared NFS
(NOT in git — `# THROWAWAY` convention); outputs under `weekend_logs/`.
Check with: `ps -eo pid,cmd | grep 'run_z_swee[p]'`,
`ls weekend_logs/z_sweep/*.json`, `cat weekend_logs/z_sweep/DONE.b05`
(+ `DONE.sparse`, `DONE.sparseT`).

**Job design:** z-window ladder {z3 = ±3 µm/13 cands, z0 = fixed} ×
{A, C} 9-rod + C3 3-clean-rod, all at `coeff_bound` 1.5 — the z0-vs-z3
contrast tests whether the per-rod z-shift is the escape channel that
drives the ±1.5 pinning (see c-sweep section below for why the data do
NOT want 1.5 waves). Each JSON records
`unexpl_frac_at_fit` (fitted model vs data, same metric as the c-sweep).

**How to read the results** (when the JSONs land — primary contrast:
C z0 vs C z3, since C's rods are clean; A is the replication with the
2 known bad-z rods; C3 = 3 clean rods, caveat: weak orientation
spread 72°/95°/178°):

| outcome | verdict | next step |
|---|---|---|
| **z0 off-bound, z3 pinned** (A and/or C) | the per-rod z-shift WAS the escape channel | ship "derive z globally, fix (or ±2σ) z" per `plans/z-range-derivation.md` O-z1/O-z3; re-run region3 (should de-pin); re-do the field test |
| **both pinned** | z is not the channel; Z5–Z8 flat valley (under-determined) is intrinsic | don't report Z5–Z8 as point estimates — upper bounds or a regularised posterior; next probes: coefficient prior (O-z4), more/better-oriented rods, then `plans/tiled-field-scan.md` |
| **z0 pins, z3 de-pins** (unexpected) | z-freedom was absorbing a real rod misplacement — check `seg_z_um`: if A's bad-z rods (r4, r6) drove it to ±3 µm, it's the rod quality | redraw the 4 bad z-clicks (A r4→z13, A r6→z18, B r5→z19, B r8→z5), re-run A z0 |
| **C3 z0 off-bound but C z0 pinned** | 9 rods / contamination / free amplitudes are needed, not z | stricter selection (isolated rods only); per-tile fits |
| **C3 z0 pinned too** | even 3 clean z-fixed rods can't de-pin → model/basis issue at this SNR | tile scan for stationarity; width/model probes; report bounds |

Also compare `unexpl_frac_at_fit` z0 vs z3: if z3's z-search barely
reduces the residual (see c-sweep: aberration itself only buys ~4 pts),
the z-moves are reshuffling the flat valley, not explaining signal.

**Probes finished 2026-11-04** (outputs in `weekend_logs/resid_probe/`,
local copies too): `run_resid_probe.py` (per-rod zero-aberration
residuals, `resid_{A,B,C}.{json,tif}`), `run_c_sweep.py`
(`csweep_{A,C}.json`), `run_width_probe.py` (see below).

**Killed 12:25 UTC on capacity grounds** (load 424 → 300): the z-reg
serial (`run_z_reg.py`: A z1 ±1 µm — 45 min of fit lost — and queued
A z05 / C z1 / C z05), one-shots A z05 / C z1 / C z05, C3 z3_b5
(bound ±5). The z-ladder middle rungs can be relaunched cheaply if the
z0-vs-z3 contrast is ambiguous. Earlier `run_fieldtest_diag_buggy_z/`
outputs and the first `c3probe/launch.log` blocks are from killed
launch attempts — provenance notes in the logs.

### Handoff checklist (for a new agent; state as of 15:50 UTC 2026-11-03)

- **Access:** from the laptop, only `nodeasb01g` is reachable (key auth
  via `~/.ssh/config`, `IdentityFile ~/.ssh/id_ed25519`, user
  `moblaende26`; GitHub token in `~/.git-credentials`). `nodeasb01` is
  reserved by a coworker / firewall-blocked — do NOT schedule there.
  Both nodes share the `/asbdata` NFS.
- **Run state:** 3 detached `nohup` processes (A 3527375, B 3527376,
  C 3526829) + a watcher, launched 15:06–15:10 UTC 2026-11-03. If this
  list is stale, re-check with `ps -eo pid,etime,cmd | grep
  run_fieldtest_di[a]g`.
- **Check progress** (one-liners from the laptop):
  ```bash
  ssh nodeasb01g 'cat /asbdata/moblaende26/zerniky/weekend_logs/fieldtest_diag/DONE 2>/dev/null || echo running'
  ssh nodeasb01g 'cd /asbdata/moblaende26/zerniky/weekend_logs/fieldtest_diag && ls *.json; for R in A B C; do printf "%s: " $R; tail -1 ${R}_diag.log; done'
  ```
- **When results exist, read them** (per JSON: `coeffs`, `pinned` at
  ±1.5, `pinned05`, `seg_z_um` — z-residuals should no longer saturate
  the ±3 µm edge; `excluded` lists the bad-z rods):
  ```bash
  ssh nodeasb01g 'cd /asbdata/moblaende26/zerniky/weekend_logs/fieldtest_diag && python3 -c "
  import json,glob
  for f in sorted(glob.glob(\"*.json\")):
      d=json.load(open(f)); c=d[\"coeffs\"]
      print(f, d[\"n_rods\"], {k:round(v,3) for k,v in c.items()}, d[\"pinned\"], d[\"seg_z_um\"])"'
  ```
- **Decision tree** (then write up + decide next steps per "Revised
  follow-up order" above):
  - off-bound + seed-stable (±~0.05 w) + coherent A/B/C pattern → z-bug
    was the pinning cause; re-do the ε/σ field comparison (original
    question answerable; `full` jobs are the ε/σ inputs, `full_clean`
    the contamination check);
  - off-bound but seed-scattered → 9-rod tiles under-determined at our
    SNR → more/wider tiles;
  - still pinned at ±1.5 (correct z, clean rods) → bound/model suspect
    revived → §5.3 probes; compare against region3 (correct z, pinned at
    ±0.5: reference case that the bug does NOT explain).
- **Resume/abort:** runner skips existing JSONs (re-run
  `.venv/bin/python -u run_fieldtest_diag.py ALL` from
  `/asbdata/moblaende26/zerniky` on nodeasb01g); the watcher writes
  `ABORTED` if all processes die early (check `watcher.log`).

### Revised follow-up order

1. Wait for the corrected 8 fits (~2–3 h).
2. **If off-bound and stable across seeds:** re-do the ε/σ field
   comparison with corrected z (the original question is finally
   answerable), then proceed to §6.
3. **If still pinned:** probe with corrected z (region3 as reference —
   correct z, still pinned), consider the bound once more, then
   capsule/width/model probes (§5.3).
4. O1 (per-rod support cropping, `plans/fit-speedups.md`) remains the next
   big perf item — it makes every iteration of 2–3 above 5–20× cheaper.

## 10. SPARSE/DEEP CAMPAIGN (2026-11-05): user-drawn regions T, S, D, E — depth ladder, sign stability, and the missing bright+deep cell

After §8/§9, the maintainer drew four new ROI sets on the full AF488 channel
(numerically verified + scored by `verify_{sparse,S,T,D,E}.py`-style scripts,
all throwaway on the server): **S** = `RoiSet_sparse` (6 rods, z 40–56 µm),
**T** = `RoiSet_sparse_high` (8 rods, z 3–7 µm, stored z accurate to ±3),
**D** = `RoiSet_deep` (6 rods, z 48–60 µm), **E** = `RoiSet_deeper` (7 rods,
z 58–67 µm). Plus a **full-field auto-scan** (`scan_sparse.py`: 3,037
segments) and a **T leave-one-out** (LOO) stability study. All fits via the
patched `run_z_sweep.py` (z0 = trust derived z, per §9.4), run on nodeasb01
(reachable again, idle) + nodeasb01g.

### 10.1 Master results table (every campaign fit)

| fit | depth (µm) | rods | bound | Z5 | Z6 | Z7 | Z8 | pinned | unexpl@fit |
|---|---|---|---|---|---|---|---|---|---|
| T z0_b05 | 3–7 | 8 | 0.5 | +0.109 | −0.120 | +0.029 | +0.004 | **no** | 0.679 |
| T z0_b15 | 3–7 | 8 | 1.5 | +0.109 | −0.120 | +0.029 | +0.004 | **no** | 0.679 |
| T b05_tsub (−r3,−r8) | 3–7 | 5 | 0.5 | +0.020 | −0.152 | +0.009 | +0.015 | no | 0.702 |
| T LOO −r1 | 3–7 | 7 | 0.5 | +0.099 | −0.131 | +0.028 | +0.002 | no | 0.681 |
| T LOO −r2 | 3–7 | 7 | 0.5 | +0.116 | −0.116 | +0.024 | +0.005 | no | 0.672 |
| T LOO −r3 | 3–7 | 7 | 0.5 | +0.075 | −0.128 | +0.022 | +0.009 | no | 0.686 |
| T LOO −r5 | 3–7 | 7 | 0.5 | +0.168 | −0.133 | +0.034 | +0.001 | no | 0.683 |
| T LOO **−r6** | 3–7 | 7 | 0.5 | **−0.187** | **+0.030** | +0.043 | +0.006 | no | 0.619 |
| T LOO −r7 | 3–7 | 7 | 0.5 | +0.139 | −0.057 | +0.025 | +0.002 | no | 0.687 |
| T LOO −r8 | 3–7 | 7 | 0.5 | +0.112 | −0.112 | +0.029 | +0.004 | no | 0.684 |
| S z0_b05 | 40–56 | 6 | 0.5 | **+0.500** | **+0.500** | +0.124 | +0.102 | **YES** | 0.559 |
| S z0_b15 | 40–56 | 6 | 1.5 | in flight (both nodes) | | | | | |
| D z0_b05 (−r5) | 48–60 | 5 | 0.5 | in flight | | | | | |
| E z0_b05 (−r2,−r3,−r7) | 58–67 | 4 | 0.5 | in flight | | | | | |
| early crop (Oct, reference) | 3.5–10 | 4 | 0.5 | +0.180 | +0.123 | +0.008 | −0.015 | no | — |
| A/B/C full (§8) | 26–65 | 7–9 | 1.5 | pinned ±1.5 | | | | yes | 0.56–0.68 |
| A/B/C b0.5 wave (z3) | 26–65 | 3–9 | 0.5 | in flight (nodeasb01g) | | | | | |
| C3 b0.5 (z3) | 51–64 | 3 | 0.5 | in flight (nodeasb01g) | | | | | |

### 10.2 T leave-one-out: the astigmatism AXIS is rod-composition-driven

Dropping **r6 alone** flips Z5's sign and rotates the axis ~90°
(+0.109/−0.120 → −0.187/+0.030); dropping r2 (the SNR-95 superstar) or r8
changes nothing; dropping r7 halves Z6's magnitude. Mechanism: r6 (161°) and
r8 (62°) sit 1.5 µm apart — an almost orthogonal pair, and an orthogonal pair
is what fixes the astigmatism axis; r6 out-signals r8 ~6×, so the axis
follows r6. The earlier `tsub` subset test did NOT contradict this — it
dropped r8 (r6's partner), which *locked* the axis onto r6.

**Consequence: never report signed Z5/Z6 from a single tile.** The stable
quantities are the astigmatism **magnitude (~0.15–0.2 w here)** and Z7/Z8 ≈ 0.
This also retroactively explains the deep-region sign flips (A −,− vs C +,+):
bound-pinning *plus* composition-driven axis — doubly uninterpretable.

### 10.3 Depth ladder: shallow de-pins, deep pins — at matched recipe

| depth | region | rods | outcome at b0.5, z0 |
|---|---|---|---|
| 3–7 µm | **T** | 8 (sparse, bright: 4 rods SNR≥12) | **off-bound, +0.11/−0.12** |
| 3.5–10 µm | early crop (Oct) | 4 (sparse, dim) | off-bound, +0.18/+0.12 |
| 40–56 µm | **S** | 6 (sparse, dim) | **PINNED ±0.5** |
| 26–65 µm | A/B/C | 7–9 (crowded) | pinned ±1.5 (b0.5 arms in flight) |

S pinned **despite** amplitudes 5× below T's (0.001–0.006) — the low-lever
regime that stayed near zero in region4/5/6 slid to the bound here, and S
explains its tubes *better* than T (unexpl 0.56 vs 0.68), so haze is not the
driver. Leading explanation: **depth-dependent spherical aberration (index
mismatch) leaking into Z5–Z8**, exactly as the maintainer hypothesised
("higher in the image is easier — probably also impacts the aberrations").
Caveats: n=1 tile per depth; brightness and field position not yet
disentangled (S is dimmer and elsewhere; the LOO shows within-tile
composition effects of the same order).

### 10.4 The bright+isolated+deep cell does not exist in this specimen

Three independent deep selections (S, D, E) plus the field scan: best deep
SNRs are 16 (S), 3.0 (D), 5.8 (E); the scan's brightest deep candidates
(z 80–120, SNR 9.5–22) all sit in dense patches. Brightness at depth comes
bundled with density (haze); isolation at depth comes bundled with dimness.
**Rod-based aberration measurement below ~30 µm is not viable in this
specimen** — no ROI-selection skill changes that.

### 10.5 Drawing at depth is intrinsically unreliable

D r5: drawn z 119, axial argmax 129 (+10). E r2/r3/r7: +10/+11/+6. At depth
the axial profile is smeared, so "sharpest plane" is ambiguous even visually
(the maintainer's own observation). Pipeline answer: always run the
z-verification (`verify_*.py`) and patch stored z to the refined argmax
before fitting (done for S; D dropped r5 instead; E fit excludes the three).

### 10.6 Field scan (for future tiles)

3,037 auto-segments; **22 bright+isolated (SNR≥8, iso≥3.5 µm) = 0.7 %**;
they form one pocket of 5 (proposal sheet + preview in
`weekend_logs/sparse_scan/proposals/`, copied to `results_local/`).
Detection is not the bottleneck — bright+isolated simply is rare, and
bright+isolated+deep does not occur (§10.4).

### 10.7 Conclusions

1. **Ruled out** as pinning causes: crowdedness (§9.2), rod quality
   (§8 full_clean), rod count (§10: 2–9 rods both pin and de-pin), z-range
   (§9.4/C3), width (§8).
2. **Shallow (≤10 µm) sparse ensembles work**: two independent tiles, a
   month apart, both off-bound at ~0.1–0.2 w magnitude. The specimen is
   near-diffraction-limited at the coverslip end.
3. **Deep (≥40 µm) sparse ensembles do not produce identifiable
   coefficients** — they pin. Depth-aberration leak is the leading
   explanation; brightness is the main residual confound.
4. **Report amplitudes, never signed Z5/Z6 axes** (§10.2).
5. **Remaining paths to a depth answer** (model-based, awaiting maintainer
   go-ahead since they touch the pinned mode set): (a) add spherical Z4 to
   the fitted modes on T (control: ~0 expected) vs S/D/E — rotationally
   symmetric, so immune to the §10.2 degeneracy, and it *measures* the
   index mismatch; (b) synthetic spherical injection into bright shallow
   data (ground truth for fitter capability).
6. **For deconvolution**: shallow slabs can use the measured near-zero PSF;
   deep slabs need either slab-wise PSFs or a model-derived spherical term —
   single-PSF-per-stack is only valid near the coverslip end.

### 10.8 In flight at documentation time

S z0_b15 (both nodes), T LOO r4, D z0_b05 (−r5), E z0_b05 (−r2,−r3,−r7)
(nodeasb01); C3 b0.5 + A/B/C crowded b0.5 arms (nodeasb01g, nice 15).
Rows to be appended on landing.

### 10.9 Compute note

nodeasb01 is reachable again and idle (the 2026-11-03 "reserved/firewall-
blocked" note in the §8 handoff checklist is stale). On the idle 256-core
node, a T-sized fit takes **9 min**; the same box projected ~1 h on the
loaded g node. `plans/fit-speedups.md` O1 remains unimplemented — the
contention behaviour (10 parallel fits thrashing a node) keeps making the
case for it.

### 10.10 Runs inventory (both nodes, 2026-11-05 ~12:00 UTC)

| node | job | state |
|---|---|---|
| nodeasb01 | D_z0_b05_nor5, E_z0_b05_sub | running (launched ~10:55) |
| nodeasb01 | S_z0_b15 (dup), T LOO r4 | running |
| nodeasb01g | S_z0_b05 (dup — will overwrite the node01 JSON with identical coeffs), S_z0_b15 | running |
| nodeasb01g | C_z3_b05_c3, A/B full_clean b0.5, C full b0.5 (nice 15) | running (~2 h 40 m) |
| both | everything else (diag, c3probe, width, scan) | DONE — see §8/§9/§10 |

---

## 1. TL;DR (ORIGINAL analysis — superseded by §8)

- **All 16 fits hit the ±0.5-wave coefficient bound** — the 15 field-test fits
  plus the region3 re-run. Nothing is a clean number anymore.
- **⚠ SUPERSEDED (§8): the field-test runner had a z-derivation bug** — every
  interior rod was placed 2–9 µm above its real depth, so the field-test
  pinning (and the "aberration ≥ 0.5 waves" conclusion below) is an artifact.
  The weekend region fits (incl. region3) used a *correct* runner and are not
  affected.
- **The guard-band hypothesis for region3 is dead** (still valid): the
  99 %-energy margin re-run pinned Z5/Z6/Z8 at −0.500 again — so region3's
  pinning is real (correct z) and points at statistics/model, not the box.
- **The A-vs-B comparison the test was designed for is impossible** with these
  results: pinned coefficients have no trustworthy sign or magnitude (a pinned
  value only says "the fit *wants* ≥ 0.5 waves in some direction").
- **What the test *can* still tell us** (§2): the aberration (or the
  model's explanation of the data) is **≥ 0.5 waves large** — larger than any
  clean weekend fit found — and the failure is **systematic, not
  field-position-specific**, because it happens everywhere: all three regions,
  all rod subsets, both seeds. That rules out "field C is special" and points
  at the fitter/model: **bound width, z-candidate range, or the
  one-shared-aberration assumption itself.**
- Next moves are §5. The decisive one is a **diagnostic re-run with wider
  bounds and a wider z range** (cheap, on the idle GPU node).

## 2. The full result table

Astigmatism axis: the angle at which the PSF is elongated; for Z5, Z6 the
axis = ½·atan2(Z6, Z5). Orientation coverage: largest angular gap between
consecutive rod orientations (0 = perfect spread over 0–180°; a gap near 180°
means all rods sit in a narrow orientation band — astigmatism **and** coma
are then barely identifiable). z-edge = rods whose fitted z-offset saturated
at the ±1.5 µm candidate limit.

| job | n | Z5, Z6, Z7, Z8 (waves) | axis | coverage gap | z-edge rods | note |
|---|---|---|---|---|---|---|
| A full s0 / s1 | 9 | +0.5, +0.5, +0.5, +0.5 | 22.5° | 33° | 2/9 | **identical both seeds** |
| A half1 | 4 | −0.5, −0.5, −0.03, −0.5 | 112.5° | 110° | 1/4 | Z7 free |
| A half2 | 5 | +0.5, +0.5, +0.5, +0.5 | 22.5° | 122° | 1/5 | |
| A pair | 2 | +0.5, +0.5, +0.5, +0.5 | 22.5° | 129° | 0/2 | 2 rods, still pinned |
| B full s0 | 9 | +0.5, **−0.5**, +0.5, **−0.5** | 157.5° | 95° | 2/9 | |
| B full s1 | 9 | **−0.5**, −0.5, +0.5, −0.5 | 112.5° | 95° | 1/9 | **Z5 sign flips between seeds** |
| B half1 | 4 | −0.5, +0.5, +0.5, −0.5 | 67.5° | 145° | 1/4 | |
| B half2 | 5 | +0.5, −0.5, +0.5, +0.5 | 157.5° | 140° | 2/5 | |
| B pair | 2 | +0.37, +0.5, +0.5, −0.5 | 27° | 98° | 1/2 | Z5 not pinned — only fit where it moved off the bound |
| C full s0 / s1 | 9 | +0.5, +0.5, **+0.001, +0.024** | 22.5° | 53° | 4/9 | **Z7/Z8 free, identical both seeds** |
| C half1 | 4 | −0.5, +0.5, −0.5, +0.5 | 67.5° | 137° | 1/4 | |
| C half2 | 5 | +0.5, +0.5, +0.01, −0.02 | 22.5° | 120° | 2/5 | **Z7/Z8 free again** |
| C pair | 2 | −0.5, −0.5, −0.5, +0.5 | 112.5° | 106° | 0/2 | |
| region3 re-run (99 %-margin box) | 7 | −0.5, −0.5, −0.04, −0.5 | — | — | 0/7 | **pins like the old run** |

The halves have terrible orientation coverage by construction (angle-split →
each half is a ~90° band, so gaps of 110–145°); the pairs were chosen for
separation, not coverage (98–129° gaps). Only the *full* fits have usable
coverage (33–53°).

## 3. What we CAN learn from this

### 3.1 The aberration (or the model miss) is ≥ 0.5 waves — big

Every fit — including 2-rod pairs, which have the least ability to "reach"
large coefficients — pushes at least one mode to the ±0.5 boundary. The clean
weekend fits (region4 +0.32, region6 +0.18, tright +0.14) were 2–3-rod fits
that apparently *could not* push that far, not evidence the aberration is
small. Physically, 0.5–1+ waves of astigmatism is entirely plausible for this
setup: sample tilt / cover-slip wedge over a 218 µm field is exactly the
kind of geometry that produces that magnitude (see
`SUPERVISOR_one_set_vs_Zmap.md` §3).

### 3.2 C shows a stable, partly-resolved structure: astigmatism without coma

Region C's `full` fit (9 rods) and its `half2` (5 rods) **both converge with
Z7, Z8 ≈ 0 (free, not pinned) and Z5 = Z6 = +0.5 (pinned), identically for
both seeds.** C's full fit has the best orientation coverage of all
(53° gap). This is a genuinely repeatable result: at the right side of the
field the data are consistent with "large astigmatism at a 22.5° axis,
essentially no coma." If C is near the field edge, that is consistent with
edge astigmatism growing toward the boundary.
**But** the Z5/Z6 axis (22.5°) is the *same* angle as A's full fit — and
B's full fits land on 112.5°/157.5° (which are 90° rotations of the same
astigmatism solution family: the axis is 2-fold degenerate per rod, and
sign-flips of (Z5,Z6) are reparameterizations). So "A and C agree, B
disagrees" is **not yet a clean x,y field-variation signal** — it could be
coverage-driven (B's 95° gap is the worst of the three full fits). That
ambiguity is exactly why the wider-bounds re-run (§5.1) is needed.

### 3.3 The z search saturates — unmodeled defocus leaks into Z5–Z8

In the C full fits, **4 of 9 rods sit at the −1.5 µm z-candidate edge**
(and in A, 2 of 9). A rod whose true depth offset exceeds ±1.5 µm cannot be
represented; the fitter compensates by spending aberration coefficients on
defocus-like blur. `SUPERVISOR_one_set_vs_Zmap.md` §6 already identifies
this as a quantifiable bias channel into astigmatism. It is active here.

### 3.4 What we CANNOT learn

- **No magnitude.** A pinned value is a lower bound only ("wants ≥ 0.5"),
  not an estimate; σ (seed stability) is meaningless when both seeds pin at
  the same boundary.
- **No signs/axis for A and B** (signs flip between seeds and between rod
  subsets *within* a region — e.g. B full s0 vs s1, C full vs C pair).
- **No ε vs σ comparison** — the test's decision rule (§1 of the protocol
  doc) cannot be evaluated. The test answered "is the field varying?"
  **not at all**; it answered "is the current fitter trustworthy at 9-rod
  scale?" — **no**.

### 3.5 Why the region3 mystery is now a *different* mystery

The weekend log hypothesized "cyclic-z wrap into the box edge" for region3's
pinning. The re-run with the proper 99 %-energy margin box pins identically
(Z5/Z6/Z8 = −0.500, Z7 = −0.04). So region3's failure is not a box artifact;
it is the same system-wide effect seen in A/B/C: **whenever a fit has enough
information to push coefficients above 0.5 waves, it hits the wall.**
region3 (7 rods, best coverage) was the first to do so.

## 4. Suspects, in order of cheapness (from code review, 2026-11-03)

1. **The ±0.5 bound is a design assumption, not a physical limit.**
   `fit.py:467` hardcodes `bounds = (-0.5, 0.5)`. The weekend "clean" fits
   may simply be under-determined fits that never reached the wall.
2. **The ±1.5 µm z-candidate window is too narrow** for rods scattered over
   ~10+ µm of depth (the C full rods span z −2 … +9 in-box). Widening to
   ±3 µm (13 candidates) costs ~2× in FFTs (§0 of `plans/fit-speedups.md`
   says the z-cand factor is exactly where O1/O2 attack it).
3. **One shared aberration over 312 px ≈ 35 µm tiles may be false** — but
   this is no longer the *primary* suspect: a wrong assumption would bias
   the *value*, it would not by itself force every subset to the bound with
   random signs.
4. Coverage: the halves/pairs were *designed* to be coverage-poor (bias
   check), so their results are expected to be wild and are not evidence
   against the model.

## 5. Follow-ups

### 5.1 Decisive diagnostic (first, cheap)

**Superseded by §8 (2026-11-03): the first launch of this diagnostic
(12:44 UTC, nodeasb01g) ran on the buggy z-derivation and was aborted after
`C_full_0` (preserved in `weekend_logs/fieldtest_diag_buggy_z/`). The
corrected relaunch (15:13 UTC, 8 fits, ±1.5 waves, 13 z-cands, full × 2 +
full_clean) is described in §8 with its design rationale. The original spec
follows, for reference.**

Re-run the 3 `full` jobs × 2 seeds on **nodeasb01g** (idle, 128 cores;
`/asbdata` shared, no re-upload) with, relative to the original runner:

- `coeff_bound = 1.5` (new `FitInput3D` field, commit `50f2aed`; the
  hardcoded ±0.5 is now the default, so existing physics tests are
  untouched; new `TestCoeffBound` guard pins this mechanism in the suite),
- `z_candidate_um = −3 … +3` (13 candidates, 0.5 µm steps),
- everything else identical to `run_fieldtest_matrix.py` (same crops, same
  derived capsules, same box rule).
- Server fitter: `zernike_test/fit.py` md5 `89df1222…` = local `sibilla`
  commit `50f2aed`; original backed up as `fit.py.bak_20261103_123221`.

Outputs: `weekend_logs/fieldtest_diag/{A,B,C}_full_{0,1}.json` + per-region
logs + `DONE` marker. Original `weekend_logs/fieldtest/` is untouched.

**Interpretation:**
- coefficients come off the bound, stable across seeds, **coherent pattern
  across A/B/C** (e.g. astigmatism growing toward the edge, coma rotating)
  → real, large, field-varying aberration → build the Z(x,y) map front-end
  (`SUPERVISOR…` §4 option B), with tile size set by the now-measured
  spatial variation.
- coefficients come off the bound but **scatter across seeds/subsets** →
  9-rod tiles are still under-determined at our SNR → more rods per tile /
  wider tiles, and the map question waits.
- still pinned at ±1.5 → the model is missing something structural
  (e.g. z-slicing, the scalar model, or the per-rod z grid) → go to 5.3.

### 5.2 Package changes to enable it (small, `feat:` on a branch)

- make the coefficient bound a `FitInput3D` field (default (−0.5, 0.5), so
  all existing tests keep pinning the physics conventions),
- `z_candidate_um` is already a field — nothing to do,
- (later) O1 per-rod support cropping from `plans/fit-speedups.md` so the
  wider z range (13 cands) doesn't multiply the ~90 % FFT cost 13/7 ≈ 1.9×.

### 5.3 If 5.1 still fails

- Verify the **derived capsules** (in every JSON) against the raw Fiji
  lines on the MIP — a systematic 1–2 px / 1-plane bias in the
  line→capsule step would look exactly like "unmodeled defocus + wrong
  shape".
- Probe the **z-slicing model**: fit a small 2-rod set with the confocal
  model *and* the widefield model, compare where the coefficients land.
- Consider the **trefoil (Z9/Z10)** question from `SUPERVISOR…` §5 — only
  after the 4-mode fit is stable.

### 5.4 Doc hygiene (this branch)

- `FIELDTEST_one_set_vs_Zmap.md` §6 → point to this doc.
- `AGENTS.md` "Known state": mark the guard-band hypothesis dead; the open
  issue is "system-wide bound-pinning — see FIELDTEST_results…".
- `plans/fit-speedups.md` §2.5: the crop-geometry question now has a
  prerequisite (stable fits) — note it.

## 6. What "done" looks like for the one-set-vs-Zmap question

1. 5.1 run finishes and is read with the rule above.
2. If coherent: measure the spatial correlation length of Z5/Z6/Z7/Z8
   across the field with overlapping tiles (the cheap experiment from
   `plans/fit-speedups.md` §2.5) — this single measurement sets the tile
   size and answers the supervisor's Q1 at the same time.
3. If incoherent: stability work (5.3 + more rods) before any field question
   is re-asked.

Until step 1, **neither "one global set" nor "build the map" is a
supported decision** — the field data do not yet speak; the fitter's
artifacts drown the signal.
