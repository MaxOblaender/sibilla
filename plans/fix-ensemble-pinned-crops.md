# Plan — fix: ensemble averages pinned crops into good ones

Status: **diagnosed 2026-11-17, not implemented.** Target branch:
`fix/ensemble-pinned-crops` off `main`.

## The failure

`tests/test_real_pipeline.py::TestFitEnsemble::test_synthetic_two_crop_ensemble`
fails on `main` and on every branch (verified 2026-11-17: fails identically
with the pre-O1 `fit.py`, 13 min run; deterministic across numpy 2.5.4/2.5.5).
It predates the fit-speedups work; `real_pipeline.py` and the test are
unchanged since the initial commit (`f4abdf7`).

## Root cause (measured, `scripts_extras/diag_ensemble.py`)

The test slices a 128×128 volume with 4 rods into two 64×64 crops but hands
**the same 4-rod skeleton to both**, re-centred. Measured per crop:

| crop | rods with ROI pixels inside | fit result | meaning |
|---|---|---|---|
| A (x 0–63) | 1 of 4 (one partial) | Z5=+0.082, Z6=−0.058, Z7=−0.004, Z8=+0.001 | **correct** (truth +0.08/−0.06) |
| B (y 64–127) | 4 skeletons, none real | Z5=+0.500, Z6=+0.338, Z7=−0.476, Z8=+0.270 | **garbage, pinned at ±0.5** |

`fit_ensemble` averages them with equal weight (`n_segments` = 4 for both,
regardless of how many rods are usable) → a good fit and a bounded garbage
fit average to a meaningless number. Note crop A alone is already correct:
the fitter and the physics are fine; the *inputs* and the *averaging policy*
are not.

## Two defects, separable

1. **Test inputs invalid** — each crop gets all rods instead of only those
   inside it; `n_segments` therefore overstates the usable evidence.
2. **No validity filter in `fit_ensemble`** — a crop pinned at the bound is
   averaged in at full weight. Given the field-test history (pinning is the
   dominant real-world failure, `FIELDTEST_results_one_set_vs_Zmap.md` §9–§10),
   this is a real design gap, not a test artifact.

## Plan (TDD, policy first — no imaging data needed)

Branch `fix/ensemble-pinned-crops` off `main` (after establishing `main` is
red on this test — first commit = the evidence, none of this is a regression
from the perf work).

1. **Write the policy tests first** (milliseconds, pure dicts):
   - one good crop + one pinned crop → result equals the good crop,
     `n_crops_rejected == 1`
   - weight by *usable* segments, not declared `n_segments`
   - all crops pinned → report the condition (no silent average)
   - a mode with a nonzero injected signal returning exactly 0.0 while
     reporting success → surfaced, not averaged silently
2. **Implement in `real_pipeline.fit_ensemble`**: reject/down-weight crops
   pinned at `coeff_bound`, weight by usable segments, record
   `n_crops_used` / `n_crops_rejected` in the report.
3. **Repair the integration test input construction**: pass each crop only
   the rods inside it; shrink it (smaller crops / fewer planes / shorter
   `max_nfev`) so it stops costing ~13 min.
4. **Verify**: full suite green on `main` afterwards; record before/after
   runtime of the integration test.

Separate from `feat/fit-speedups` — do not fold in there (unreadable diff,
different topic).