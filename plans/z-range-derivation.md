# Plan — Per-rod z placement: how to determine the z search range

Status: **design only.** Opened 2026-11-04 from the field-test pinning
investigation (see `FIELDTEST_results_one_set_vs_Zmap.md` §8 and the
c-sweep / z-sweep probes on `nodeasb01g`). This is the *prerequisite*
document for the "fix z, de-pin the fit" step that the z-sweep decides.

## 0. The problem this answers

If the z-sweep (running 2026-11-04) confirms **Outcome 1** — i.e. closing
the per-rod z-window (z3 → z0) pulls the Z5–Z8 off the ±1.5 bound — then
the fitter's correctness becomes *dependent on the z we hand it*. A fixed
z that is wrong is a silently-wrong aberration. So before we ship "fix z"
we need a principled, **image-agnostic** way to place each rod in z.
The question: **given a new image, how do we determine each rod's z and
how much local search (if any) is enough?**

Key fact established this session: a ~2 µm z-error ≈ ~0.2–0.3 waves RMS
of defocus-equivalent spread across the 9-rod tiles — **the same order as
the aberration we are trying to measure** (clean weekend fits were
0.14–0.32 waves). So z-placement is not a small correction; it is a
confound of comparable magnitude to the signal.

## 1. Two different "z variance" notions — don't conflate them

| notion | what it is | does it break fixed-z? |
|---|---|---|
| **Rod-to-rod z spread** | the bacteria sit at different heights in the sample | **No** — each rod carries its *own* derived z (per-endpoint). High spread is handled. |
| **Per-rod z uncertainty** | how sure we are of *this* rod's depth | **Yes** — fixed z assumes the derivation is right. High uncertainty ⇒ the z we fixed is wrong. |

The danger is the second one. "An image has higher z variance" is only
harmful if it means higher **per-rod uncertainty**, not more spread.

## 2. How z is derived today (the baseline, already image-agnostic)

`run_fieldtest_diag.derive()` / `server_fit.py` (the *correct* one):

1. **Start plane** = the z the user clicked in the ROI (Fiji `.roi`,
   byte 58). No global re-search at this stage.
2. **xy refine** ±3 px toward the brightest pixel (perpendicular to the
   line) — lateral only.
3. **Per-endpoint local z search**: ±1.5 µm window (7 planes, 0.5 µm step)
   at each capsule end; pick the plane with the highest 5×5-median
   intensity (`find_z`).

This is already per-image (uses the data's own intensity) and per-endpoint
(handles a rod tilted in z). The z-sweep's `z0` rung fixes exactly this
value. **So "fixed z" ≠ "the clicked z" — it means "the derived profile
peak z."** The derivation is the thing that must be robust.

## 3. Design options for a robust per-rod z (ranked)

### O-z1 — Derive from the *global* axial profile, not a window  *(recommended first)*

Replace the "max within ±1.5 µm of the click" with "**global argmax of the
rod's axial intensity profile**" at its xy. Then the ±1.5 µm *window* is
unnecessary — the constraint is "one rod, one depth, decided by data."

- **Fixes** the exact failure mode we found: a click on a brighter
  neighbouring rod, where the local search *confirms* the wrong plane.
  A global peak picks the rod's own in-focus plane even if the click was
  off.
- **Still fooled by** a laterally-overlapping rod within ~1–2 px (the
  profile then has two peaks). → pair with O-z2 to detect that.
- **Cost**: negligible (one extra profile slice, already computed).
- **Gate**: equivalence — on the 27 field-test rods, global-argmax z
  should equal the known-good peak within ±1 plane; the 4 bad rods
  (A r4/r6, B r5/r8) should be *reassigned to the correct brighter rod*
  (that is the win, not a regression).

### O-z2 — Fit the axial profile, report a confidence  *(recommended, adds the uncertainty model)*

Fit a 1-D profile (Gaussian / Lorentzian / sum of two Gaussians) to the
axial intensity at the rod xy, and output:
- `z_peak` (sub-plane), `sigma_z` (width → uncertainty), and
- **a doublet flag**: if a 2-Gaussian fit is markedly better than a 1-G
  (BIC/ΔAIC) → the column contains **two rods in z** → ambiguous.

This turns "high per-rod z uncertainty" (the harmful variance) into a
*reported number* the fit can act on, and it flags the laterally-overlapping
case O-z1 can't see.

### O-z3 — Search window scaled to the derived uncertainty  *(replaces a global constant)*

Do **not** ship a fixed ±3 µm or ±0.5 µm. Ship:
```
z_candidate_um = {0}                              if profile is sharp (single peak, low sigma)
                = {-2σ, -σ, 0, +σ, +2σ}           if profile is broad / doublet
```
i.e. "mostly fixed, small local freedom" sized to *that rod's* profile.
On a new image with genuinely more z variance, **the data widen the search
themselves** — no code change, no magic number. This is `z_candidate_um`
as a derived field, not a constant. (The ±0.5 µm / 3-candidate rung already
exists in the fitter as the cheap path.)

### O-z4 — Per-rod z as a *soft* nuisance (the modelling fix) *(defer, larger)*

Rather than discrete candidates, treat each rod's z as a continuous
parameter with a penalty pulling it to its derived value:
`cost += (λ_z / σ_z²)·Σ_s (z_s − z_s^derived)²`. A *small* λ keeps z near
the derivation (de-pinning) while allowing it to move where the data
really disagree. This is the proper "regularised z" and removes the
discrete-greedy cross-talk between co-located rods found this session
(A r3 & r9, B r6 & r8). **Bigger change** (rework of the alternating
c/z loop), so defer until O-z1..z3 + the sweep confirm the direction.

## 4. Decision the running sweep must feed into this

The z-sweep (2026-11-04, A/C at z3/z0, C3-3rod at z3/z0) resolves:

- **z0 de-pins, z3 pins** → O-z1 + O-z3 are sufficient; ship "derive
  globally, fixed (or ±2σ) z". The fitter no longer needs a wide search.
- **both pin** → z is *not* the escape channel; z placement is a red
  herring and the real issue is the flat Z5–Z8 valley (see
  `fit-speedups.md` §2.5 + the c-sweep: aberration has no explanatory
  power). Then O-z4 / a coefficient prior is the next thing to try, and
  the z range question is moot (leave it at the derived value).

**Until that result is in, do not commit a z-range default.** This plan is
the design; the sweep is the gate (consistent with `fit-speedups.md` §3:
every change is a flag + equivalence test, never a silent default).

## 5. What "done" looks like

1. z-sweep read out → z0-vs-z3 verdict recorded here + in
   `FIELDTEST_results…md` §8.
2. If z0 wins: implement O-z1 (global-argmax derivation) as a flag with an
   equivalence test (27-rod reassignment check above); add O-z2
   `sigma_z` + doublet flag to the derivation output (small, testable).
3. `z_candidate_um` becomes derived per rod (O-z3); the wide ±3 µm is
   documented as *not* a default.
4. Re-run region3 + the A/B/C field test at the new default; confirm
   de-pinning and a meaningful ε/σ.
5. Only then: consider O-z4 (soft z nuisance) as a `perf:`/`fix:` commit.
