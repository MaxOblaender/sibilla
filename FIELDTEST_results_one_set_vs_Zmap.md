# Field-test results — what the A/B/C fits tell us (2026-11-03)

Companion to `FIELDTEST_one_set_vs_Zmap.md` (protocol, §6 results now filled by
this doc). Status: **2026-11-03 UPDATE: a z-derivation bug in the field-test
runner was found and fixed; ALL 15 field-test fits are confounded and the
corrected 8-fit run is in progress (§8).** Everything below §8 is the
original analysis — kept for the record, but its main conclusion (large
aberration) is retracted.

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

### Progress snapshot (2026-11-04 07:40 UTC) — 4/8 fits done, all still pinned

C is complete; A and B are on their second full seed (wall ≈ 9.5 h per
full fit, so expect A/B seed1 + full_clean during the day of 2026-11-04):

| job | Z5, Z6, Z7, Z8 (waves) | pinned / pinned05 | z-residuals (µm) |
|---|---|---|---|
| C full s0 (175 min) | +1.5, +1.5, +0.011, +0.076 | yes / yes | ±3.0 on 2/9 rods (r8, r9) |
| C full s1 (188 min) | +1.5, +1.5, +0.011, +0.076 | yes / yes | identical to s0 (exact) |
| A full s0 (568 min) | −1.5, −1.5, +0.821, −0.140 | yes / yes | ±3.0 on 3/9 rods; bad-z r6 at 2.5 |
| B full s0 (548 min) | −0.039, +1.5, +1.5, −0.389 | yes / yes | ±3.0 on 1/9 rods (bad-z r5); r7 at 2.5 |
| A full s1, A full_clean, B full s1, B full_clean | — running — | | |

Interim read (per the decision tree below): C's **seeds agree exactly**
(seed stability is good), but the fits do **not** come off the ±1.5 bound
with corrected z — and A/B's full_0 still carry the 2 bad-z rods each
(the full_clean jobs are the contamination check). Since region3 (correct
z, clean rods) pinned at ±0.5 too, the "bound/model suspect" branch stays
alive. The z-residuals no longer saturate uniformly (3 of 9 rods are at
the ±3 µm edge, not all 9), which is consistent with the z bug being gone;
what remains is the bound/model question.

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
