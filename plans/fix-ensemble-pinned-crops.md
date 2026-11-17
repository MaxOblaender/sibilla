# Plan — fix: ensemble averages pinned crops into good ones

Status: **implemented on this branch, validated (policy 7/7, wiring 2/2,
integration 1/1 + full test_real_pipeline 10/10 on nodeasb01g), awaiting
merge decision.** Adversarially reviewed 2026-11-17 (mutation-proven tests;
per-crop convergence 0.0003 waves). Target: merge to `main` first, then
`feat/fit-speedups`.

## Root cause (measured; corrected by adversarial review)

**Three defects, in layering order:**

0. **Key-type bug in the test itself (the direct cause of the red suite):**
   `fit_ensemble`'s report serializes modes as STRING keys
   (`{"5": 0.08, ...}`) but the test read `res["coeffs"].get(5, 0.0)` with an
   INT key — silently `0.0` every time. The assertion therefore evaluated
   `|0.0 − 0.08| < 0.04` → **False regardless of fit quality**. Verified by
   adversarial review: the mismatch pre-existed at HEAD, so the old test was
   unpassable on key grounds alone. This is why the reported error was
   exactly 0.08 (and why it looked like a fit stall — it wasn't one).
1. **Invalid test inputs:** one 128² volume sliced into two 64² crops with
   the FULL 4-rod skeleton handed to both. Crop A: 1 usable rod (still fit
   correctly: Z5=+0.082, Z6=−0.058 vs truth +0.08/−0.06). Crop B: 4
   skeletons, none real → fit pinned at ±0.5. (Also structural: a 64² crop
   cannot contain two 3 µm rods with cyclic-wrap-safe margin — the sliced
   construction tests an invalid use case; real workflows fit clean crops.)
2. **No validity filter in `fit_ensemble`:** crop B's pinned result was
   averaged into crop A's good one at equal weight → the reported ensemble
   coefficients would have been ≈(0.29, 0.14) — garbage — had the key bug not
   masked it. Field-test context: pinning is the dominant real-world failure
   (`FIELDTEST_results_one_set_vs_Zmap.md` §9–§10), so this is a real design
   gap, not a test artifact.

## Plan (TDD, policy first — no imaging data needed) — ALL DONE 2026-11-17

1. ~~Policy tests~~ **done**: `tests/test_ensemble_policy.py` — 7 tests
   (exclusion, weight respect, near-bound admissible, all-pinned raise,
   majority-pinned, empty-input raise, zero-weight raise). Mutation-proven:
   reverting to naive averaging fails exactly the 3 pinned-crop tests.
2. ~~Implement~~ **done**: `real_pipeline.combine_ensemble` +
   `fit_ensemble` rewired; bootstrap and λ-sensitivity refits restricted to
   admissible crops; report gains `n_crops_used`/`n_crops_rejected`/
   `used_crops`/`rejected_crops`.
3. ~~Repair the integration test~~ **done**: two INDEPENDENT valid crops
   (different seeds, same aberration) instead of one sliced volume; string
   report keys; plus 2 wiring tests (monkeypatched `fit_3d`: pinned-crop
   exclusion and all-pinned propagation through `fit_ensemble` — closes the
   gap found by adversarial review).
4. ~~Validate~~ **done**: policy 7/7 (0.5 s), wiring 2/2, integration 1/1
   (per-crop convergence 0.0003 waves, both crops), full
   `test_real_pipeline.py` 10/10 on nodeasb01g (2:54). Full-suite run on the
   server: see branch status before merge.

Note: weight semantics stay "declared n_segments" (the repaired test declares
honestly); a usable-segment recount inside `fit_ensemble` is a possible
follow-up, not needed for this fix. Known quirk left as-is: mode set comes
from the first admissible crop (`used[0]["coeffs"]`) — all callers use
REAL_MODES.