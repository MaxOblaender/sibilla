# Field-test results — what the A/B/C fits tell us (2026-11-03)

Companion to `FIELDTEST_one_set_vs_Zmap.md` (protocol, §6 results now filled by
this doc). Status: **data complete, interpretation + follow-ups.** All numbers
read from the 15 JSONs in `weekend_logs/fieldtest/` on the shared server
(`/asbdata/moblaende26/zerniky/`), plus the region3 re-run
(`weekend_logs/rerun_region3.log` + `fit_region3.json`).

Fitter: server copy `zernike_test` (`fit.py` md5
`4fcb3e0c8b910f6677a603b054b3905e`, **same commit as this branch**),
`fit_3d`, confocal model, 7 z-candidates (−1.5 … +1.5 µm, 0.5 µm steps),
`n_multi_start=2`, `max_nfev=150`.

---

## 1. TL;DR

- **All 16 fits hit the ±0.5-wave coefficient bound** — the 15 field-test fits
  plus the region3 re-run. Nothing is a clean number anymore.
- **The guard-band hypothesis for region3 is dead.** The re-run with the
  99 %-energy margin box (36×393×393, 7 rods, 1313 min, finished 06:15 UTC
  2026-11-03) pinned Z5/Z6/Z8 at −0.500 again. Bigger box did not help →
  the cause is not cyclic-z wrap at the box edge.
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

Re-run the 3 `full` jobs × 2 seeds on **nodeasb01g** (idle, 128 cores;
`/asbdata` shared, no re-upload) with, relative to the original runner:

- `bounds = (−1.5, 1.5)` waves (temporarily; make configurable in-package
  first — see 5.2),
- `z_candidate_um = −3 … +3` (13 candidates, 0.5 µm steps),
- everything else identical (same crops, same derived capsules, same box
  rule) so the only differences are the two limits.

6 fits × ~50 min ≈ 8 h wall if serialized — but they are independent, so
~3 in parallel → **~2.5 h**, `nohup`'d, commit hash logged.

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
