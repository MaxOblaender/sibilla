# Field-test protocol & result — does the aberration depend on image position?

Status: **runs launched 2026-11-02 evening, results pending (§6).**
This doc records what was done, what runs overnight, and — once finished —
what the data show. Decision it feeds: one global Z5–Z8 vs. a Z(x,y) map
(see `SUPERVISOR_one_set_vs_Zmap.md` §4/§7).

## 1. Question

Is a *single* set of Z5–Z8 valid for the whole AF488 channel
(218 × 163 µm, 98 µm z), or does the aberration vary with field position
(enough that one global value is wrong beyond its uncertainty)?

The test measures **ε** (variation of the fitted Z between positions) against
**σ** (uncertainty within one position, from repeated/seeded fits):

| outcome | interpretation |
|---|---|
| \|Zᵢ − Zⱼ\| ≲ 2σ for all region pairs | field variation below accuracy → one global set is fine |
| \|ΔZ\| ≫ σ **and** coherent pattern (coma radial, astig ∝h²) | real field aberration → need Z(x,y) |
| large \|ΔZ\|, **no** coherent pattern / changes between repeats | fit instability or model bias → fix the fit first |

## 2. Input (manual, 2026-11-02)

Three regions picked from a full-channel signal-density scan
(`block_scan.py` on the server; MIP figure `results/block_scan_mip.png`),
chosen for spatial separation and comparable rod depth:

| region | field (µm) | crop z (planes / µm) |
|---|---|---|
| **A** | x 98–134, y 16–52 (left-center) | 24–42 |
| **B** | x 112–147, y 98–134 (center) | 30–48 |
| **C** | x 180–215, y 30–65 (right, near edge) | 48–66 |

Caveats: C sits ~20 µm deeper than A/B → C-vs-A/B differences cannot
separate x-position from depth (RI mismatch) — **A vs. B is the clean
x,y comparison.** C is near the field edge (edge effects expected).

Crops (36 × 312 × 312) on the server as `fieldtest_{A,B,C}.tif`, synced to
`fieldtest_crops/` on the laptop. **9 line ROIs per region** drawn by hand in
Fiji (isolated rods, varied orientations), saved as
`fieldtest_{A,B,C}_rois.zip`.

**Fiji `.roi` layout note (new, 2026-11-02):** the *new* Fiji files store the
z-plane as int16 at **byte 58** (byte 74 = 0); the old `RoiSet.zip` used byte 74.
Cross-checked against the intensity z-peak: 23/27 lines agree within ±3 planes,
the rest differ because the clicked plane is one rod end and the sharpest end
is the other. Runner reads 58 with 74 as fallback. **Update AGENTS.md's
Fiji-ROI note once verified with a second Fiji version.**

## 3. What runs overnight (throwaway, not in the repo)

Server scripts `/asbdata/moblaende26/zerniky/`:
- `run_fieldtest_matrix.py` — per region:
  - `full` × seeds 0 and 1 (σ from seed stability),
  - `half1` / `half2` — angle-split halves of the rod set (rod-count +
    orientation-coverage bias check),
  - `pair` — the best-separated 2-rod pair (minimum-identifiability check).
- `run_fieldtest_all.sh` — 3 regions in parallel, 4 OMP threads each.

Per fit (existing `fit_3d`, confocal model, 7 z-candidates, max_nfev=150,
n_multi_start=2 — the weekend parameters):
- box = rods + 64 px XY guard (99 %-energy margin range), z = full 36-plane
  crop **extended ±10 planes from the full channel** (clamped), so cyclic-z
  wrap cannot land a rod's tail on the box edge;
- output JSON per job: `weekend_logs/fieldtest/{R}_{job}_{seed}.json` with
  coeffs, cost, strehl, per-rod z residuals, pinned-at-bound flag, **and the
  derived capsule endpoints** (for eyeball-checking the ROI→capsule step).

Log: `weekend_logs/fieldtest/{R}.log`; completion marker:
`weekend_logs/fieldtest/DONE`.

Known quirks found during setup (2026-11-02):
- New Fiji `.roi` layout: z-plane at byte 58 (old `RoiSet.zip`: byte 74).
  Runner reads 58, falls back to 74.
- Channel TIF pages are 2D (y,x); per-plane slicing is `[y0:y1, x0:x1]`.
- Zeros-pad the box if the z-extend hits the channel edge (clamped 0/196).

## 4. How to read the results

1. **Sanity:** derived capsules vs. the drawn lines (in the JSON); z
   residuals within ±1.5 µm; no `pinned=true`; per-rod amplitudes ~1 (not in
   JSON — check cost vs. the weekend fits if in doubt).
2. **σ:** spread of Z5..Z8 between `full_0` and `full_1`, and between
   `half1`/`half2` (halves also test orientation-coverage sensitivity).
3. **ε:** region-vs-region difference of the `full` fits.
4. **Pattern (if ε ≫ σ):** coma (Z7,Z8) vector direction per region —
   expected radial (pointing away from field centre); astigmatism (Z5,Z6)
   — expected ∝h² with rotating axis. A coherent pattern = field aberration;
   an incoherent one = fit/model problem.
5. **pair job:** if a 2-rod fit already deviates from its region's `full`
   fit by ≫ σ, the mode set is under-determined at 2 rods.

## 5. Launch record

- Launched: 2026-11-02 ~18:05 (server time), `nohup ./run_fieldtest_all.sh`.
  Region C needed one extra bugfix (z-search window at crop edge) and was
  relaunched manually ~18:20; it joined the same log/JSON layout.
- Dry run before launch: `run_fieldtest_matrix.py A DRY` → box 36×169×169,
  0.97 min, plumbing OK (fake 4-line zip).
- Machine: nodeasb01 (256 cores; ~3 cores/region used → 12 of 256).
- Processes are detached (`nohup`/`setsid`) — a laptop shutdown does not
  affect them.
- Resumable: finished jobs are skipped on re-run
  (`nohup ./run_fieldtest_all.sh` again picks up unfinished jobs).

## 5.1 Where the results are (check from the laptop)

All on the server (not mounted locally — use ssh):

| what | path |
|---|---|
| one JSON per fit (coeffs, cost, strehl, z-residuals, pinned flag, derived capsules) | `/asbdata/moblaende26/zerniky/weekend_logs/fieldtest/{A,B,C}_{full,full_1,half1,half2,pair}_{seed}.json` |
| per-region log (derived lines, then one line per finished fit) | `/asbdata/moblaende26/zerniky/weekend_logs/fieldtest/{A,B,C}.log` |
| completion marker (written when all 3 regions are done) | `/asbdata/moblaende26/zerniky/weekend_logs/fieldtest/DONE` |
| launch log | `/asbdata/moblaende26/zerniky/weekend_logs/fieldtest/launch.log` |

One-liners from the laptop:

```bash
# is it done?
ssh nodeasb01 'cat /asbdata/moblaende26/zerniky/weekend_logs/fieldtest/DONE 2>/dev/null || echo still running'

# progress (last fit result per region)
ssh nodeasb01 'tail -2 /asbdata/moblaende26/zerniky/weekend_logs/fieldtest/{A,B,C}.log'

# the still-running region3 rerun (see below)
ssh nodeasb01 'ps -o pid,etime,cmd -p 3703039 2>/dev/null || echo region3-finished; tail -1 /asbdata/moblaende26/zerniky/weekend_logs/rerun_region3.log'

# pull all results to the laptop
rsync -av nodeasb01:/asbdata/moblaende26/zerniky/weekend_logs/fieldtest/ \
    ~/Documents/git_repos/sibilla/weekend_logs/
```

Expected: 5 fits per region (full × seeds 0,1; half1; half2; pair) = 15
JSONs. `full` fits are the ones that matter for ε; the rest give σ and
bias checks. If fewer JSONs exist, check the corresponding `{R}.log` for
tracebacks and re-run `./run_fieldtest_all.sh` (resumable).

**Also still running: the region3 rerun** (started 2026-10-30, box
36×393×393, 7 rods, 99 %-margin — the one that last pinned Z5/Z6/Z8 at
−0.5). PID 3703039, ~8 h 19 min as of 2026-11-02 18:25, no result yet.
Check it with:

```bash
ssh nodeasb01 'ps -o pid,etime,cputime,cmd -p 3703039 2>/dev/null || echo region3 finished
tail -3 /asbdata/moblaende26/zerniky/weekend_logs/rerun_region3.log'
```

When it finishes it writes `weekend_logs/fit_region3.json` (coeffs +
per-rod z residual). **Caveat:** the file that exists on disk right now
is the *old* Oct-30 result (376 min, pre-guard 320² box, pinned at −0.5)
— the rerun overwrites it, so read it only after `region3 finished` or
after the JSON's mtime is newer than the fieldtest launch. If it pins at
the bound again with the 99 %-margin box, the box-guard hypothesis for
the region3 failure is dead and the cause is in the statistics /
field-variation side (this test).

## 6. Results

All 15 fits finished (DONE marker 2026-11-02 20:27 CET); the region3
99 %-margin re-run finished 2026-11-03 06:15 UTC.
**Summary: all 16 fits pinned at the ±0.5-wave bound; the guard-band
hypothesis for region3 is dead; the A-vs-B comparison is impossible with
pinned values. Full analysis and the follow-up plan:**
`FIELDTEST_results_one_set_vs_Zmap.md` (2026-11-03).

## 7. Follow-ups (decision-dependent)

- If ε ≲ σ: proceed with **one global set** design; the per-rod support
  crop + parallel FFT is the speed-up (see `plans/fit-speedups.md`).
- If ε ≫ σ with pattern: design the **Z(x,y) map** front-end (tiling +
  combine step); also ask supervisor about a parametric field model
  (`SUPERVISOR_one_set_vs_Zmap.md` §7 Q2).
- If incoherent: debug fit stability first (z-nuisance bias, guard band,
  multi-start) before any field question.
