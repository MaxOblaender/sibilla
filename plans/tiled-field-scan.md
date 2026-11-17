# Plan — Tiled field scan: is the aberration position-dependent?

Status: **design only, nothing implemented.** Drafted 2026-11-04 during the
z-sweep wait. This is the *next* decisive experiment after the z-sweep,
and it is the direct measurement behind the one-set-vs-`Z(x,y)`-map
decision in `FIELDTEST_one_set_vs_Zmap.md`.

## 0. Why this experiment exists (motivation from 2026-11-04)

1. **The fitter fits ONE shared Z5–Z8 per box.** If coma/astigmatism
   actually vary across the ~35 µm field, a shared value is the average
   of a varying quantity — biased, and it *flattens the cost surface*
   (any single value is as good as another → optimizer wanders to the
   bound). Field variation is therefore a candidate *cause of the
   pinning itself*, not just a nuisance afterwards.
2. **The C3 3-rod probe was the wrong design for this question** (my
   error, recorded here so it isn't repeated): its rods were chosen for
   zero tube contamination and ended up spread across ~26 µm of field
   (r2 y≈52, r8 y≈107, r9 y≈279). If the field varies, fitting one
   coefficient set to three differently-aberrated rods is guaranteed
   to be a compromise — a "C3 pins" result would be uninterpretable.
   Tile-based fits fix this by keeping each fit spatially compact.
3. The supervisor decision (one set vs `Z(x,y)` map) needs the
   **correlation length** of the aberration field; nothing so far
   measures it.

## 1. Design

- **Grid:** 3×3 tiles, side ~10–12 µm (~90–110 px), 50 % overlap,
  centred on the 312 px crop → 9 tiles (extend to 4×4 = 16 if the
  first pass shows structure).
- **Rod criteria per tile** (from the 2026-11-04 rod-quality audit,
  `rod_quality_report.py`): `OK` verdict — |dz| ≤ 2 planes, lateral
  ≤ 8 px, tube contamination < 10 %, plus **nearest tube ≥ 2 µm away**
  (outside the 1.5 µm ROI column). 2–4 rods per tile.
- **Orientation coverage:** ≥ 3 distinct angles per tile, worst gap
  ≲ 80° (the C3 set's 72°/95°/178° is the counter-example). If a tile
  can't meet this, record it but fit anyway — flag the result as
  weakly identified.
- **Fit configuration:** z **fixed** at the derived profile peak
  (`z0` rung, 1 candidate), `coeff_bound` 1.5, confocal, same box rule
  as the field test. Cost per tile is tiny: box volume scales as
  (100/312)² ≈ 0.1× → minutes per tile even single-threaded.
- **Overlap consistency:** adjacent tiles share rods (50 % overlap).
  A rod fitted in two tiles must give the same local contribution —
  discrepancies flag non-stationarity *within* the tile scale.

## 2. Controls (what makes the result trustworthy)

| control | purpose |
|---|---|
| Same rods, same box, z fixed everywhere | coefficient differences between tiles = position, not rod set |
| Overlapping tiles share ≥ 1 rod | internal consistency check |
| Empty-tile negative control (a tile with no OK rods is skipped, not padded with bad rods) | avoids the known contamination bias |
| Repeat one tile with a different seed | ties down the (weak) seed stability; expect bit-identical at z0 since z is frozen |

## 3. Readout

1. Per-tile coefficients → (Z5, Z6, Z7, Z8) at each tile centre →
   four maps over the field.
2. **Correlation length:** pairwise correlation of tile coefficients
   vs tile separation (the overlaps give short baselines). `L_c` =
   separation where correlation drops to 1/e.
3. **Verdict:**
   - All tiles agree within noise → field is stationary at this scale
     → **one set per tile image is justified**; the flat-valley cause
     must be sought elsewhere (model, SNR).
   - Systematic pattern (coma rotating outward, astigmatism growing
     with radius, focus tilt across the field) → **`Z(x,y)` map
     needed**; also check whether per-tile fits come *off* the ±1.5
     bound — if they do, field-averaging was a (or the) pinning cause.
   - Tile values scatter with no pattern → aberration not identifiable
     at this SNR per tile (few rods) → report upper bounds, not maps.

## 4. Prerequisites and cost

- **Gate: the z-sweep verdict.** If z3 (±3 µm) is the escape channel,
  tile fits must also fix z (or use the derived ±2σ window from
  `plans/z-range-derivation.md` O-z3), or they inherit the pinning and
  the maps are meaningless.
- z0 config: ~1.3 CPU-h for a 9-rod *full* box; a 3-rod 100 px tile is
  ~0.1× of that → **the whole 3×3 scan is well under an hour**, even
  single-threaded, and can run on nodeasb01g alongside anything.
- Data: already on the server (`fieldtest_{A,B,C}.tif` + ROI zips);
  no new imaging needed. Re-runs on other channels later reuse
  everything.

## 5. What "done" looks like

1. z-sweep read → configuration frozen (z fixed / ±2σ).
2. `scripts_extras/` (or package flag) gains a tiling helper: crop
   tiles, run rod-quality prefilter per tile, emit per-tile
   `FitInput3D`s, collect JSONs.
3. One 3×3 scan on C (cleanest rods) + one on A (has the bad-z rods,
   now avoidable by the prefilter).
4. Maps + correlation length recorded in
   `FIELDTEST_results_one_set_vs_Zmap.md`; the one-set-vs-Zmap decision
   is then a measurement, not an opinion.
