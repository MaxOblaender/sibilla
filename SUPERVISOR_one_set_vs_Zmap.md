# Open question: one global Zernike set, or a position-dependent Z(x, y)?

Draft for discussion (2026-11-02). SIBILLA project — blind Zernike aberration
estimation from our own sample images (V. cholerae, no bead calibration),
so we can deconvolve the full dataset with a measured PSF.

---

## 1. What the tool does (1 minute)

We image fixed *V. cholerae* cells — thin rods of known width (~0.6 µm) and
known positions. Each rod in the image is approximately
`rod footprint ⊛ PSF + background`. We render the rod footprint, parameterize
the PSF by Zernike coefficients (scalar, per-defocus), and fit the
coefficients that best reproduce the images. Today we fit **Z5–Z8 only
(astigmatism ×2, coma ×2)**; these four are the *differential* modes — the
ones that change the *shape* of the PSF rather than just its focus or its
overall width. The same four modes are shared by **all** rods in one fit
(the ensemble principle: many rods → one system of equations). The result
is a fitted PSF volume, which we then use to deconvolve the whole data set.

## 2. What we have measured so far

The AF488 channel is 196 z-planes × 1440 × 1920 px = **163 × 218 µm field,
98 µm in z** (0.1135 µm/px, 0.5 µm z-step; 40×/1.2 water, spinning disk).
The full field cannot be fitted as one box (single fit ≈ 80 h by measured
rates, and it would force one aberration on the whole field — see below).
So we have been fitting **regions** of the field, each with the rods the
user picked (Fiji line ROIs):

| region (field position) | rods | fitted Z5, Z6, Z7, Z8 (RMS waves) | notes |
|---|---|---|---|
| region4 (upper left) | 2 | +0.32, −0.08, +0.06, +0.02 | clean fit |
| region6 (left middle) | 2 | +0.18, −0.15, −0.06, −0.06 | |
| region5 (far left, 2 rods ~20 µm apart) | 2 | +0.09, +0.20, +0.01, −0.02 | |
| tright (right edge) | 3 | +0.14, −0.16, +0.05, −0.04 | |
| merged (5 rods spread over ~50 µm) | 5 | −0.22, +0.31, +0.01, −0.05 | |
| region3 (7 rods over ~30 × 25 µm) | 7 | **−0.50, −0.50, −0.04, −0.50** | pinned at the ±0.5-wave fit bound → result unusable |

Two observations:

1. The regional astigmatism/coma values **do not agree with each other**
   (Z5 spans +0.32 … −0.50). Either the aberration really differs across
   the field, or the single-region fits are too unstable to carry that
   conclusion (each has only 2–7 rods), or both. We cannot distinguish the
   two cases with the current data — that is the first thing to resolve.
2. The larger, more rod-dense region (region3) is the *worst* one — the fit
   hits the coefficient bound and is unusable. That is the opposite of what
   we would expect if a shared aberration were simply harder to measure with
   more rods; it suggests the region is averaging over at least two different
   local aberrations (or has a modeling problem we have not yet isolated).

## 3. The physical question

A 40×/1.2 water objective is designed to be well-corrected over a finite
field, but "field" in the design sense usually means a corrected field
diameter (often ~8–9 mm); **our 218 µm field is far inside that**. Within a
single *well-corrected* field one would expect the objective's intrinsic
astigmatism to be roughly constant in position. So a real position
dependence would come from one of:

- **Sample/alignment geometry.** If the sample plane is not parallel to the
  objective's field (tilt), the local defocus varies across the field, and
  a through-focus fit with a *shared* PSF can trade defocus-like error for
  astigmatism-like coefficients. Our model already gives each rod its own z
  position (per-rod z candidate, 0.5 µm grid), which should absorb a slow
  tilt — but only if the tilt is smooth and the z search is accurate.
- **Refractive-index mismatch along the light path.** The objective is
  corrected for water (n = 1.33). If the cells sit in a medium of slightly
  different n, or if there is a dry gap/cover effect, **spherical aberration
  accumulates with optical path length**. Spherical aberration is *not*
  position-dependent but is *depth*-dependent — and our sample spans 98 µm
  in z, so the *same rod at two depths* sees a different PSF. That is
  physically real and not a fitting artifact.
- **Genuine field-dependent astigmatism** of the objective or of the
  spinning-disk scan geometry (disk/scan-plane alignment, pinhole
  symmetry). Spinning disk can introduce position- and orientation-dependent
  blur that a static Zernike model does not capture.

**Which of these is true is a measurement question, not something I can
decide from the model.** But the answer decides the tool's architecture:

## 4. The decision: one set vs. a map

### Option A — one global Z5–Z8 for the whole channel

- Simple; one PSF deconvolves everything.
- Only valid if the aberration (astigmatism + coma) is actually constant
  over 163 × 218 µm. The regional table above is *not* consistent with that
  (but see §2.2: the regions are not independent).
- If the field really varies, Option A does not fail loudly: it fits some
  weighted average and **silently produces a PSF that is wrong everywhere,
  on average everywhere** — the deconvolution then looks plausible but is
  biased.
- Cost: with per-rod support crops (the planned speed-up), one global fit
  over all picked rods in the field is feasible (a few hours on the CPU
  server, minutes with the GPU FFT batch).

### Option B — a position map Z(x, y): fit tiles, report the field

- Correct by construction if the aberration varies; also gives us a direct
  *picture* of how the aberration varies (a strong diagnostic: the spatial
  pattern of Z5/Z6 would distinguish sample tilt from objective field
  astigmatism from disk artifacts).
- Cost: a tile grid. With the planned speed-ups, tiles of ~15–25 µm with
  2–8 rods each are cheap per tile and embarrassingly parallel (256 cores
  on nodeasb01, or the 4× A40 GPU box). The whole map would take
  roughly one long weekend of parallel CPU time with the *current* code —
  and tens of minutes after the planned speed-ups, because tile fits run
  concurrently.
- Requirements per tile: enough rods, enough orientation spread, and the
  tile must be small enough that "the aberration is constant inside it"
  holds. The tile size is a trade-off between *statistics* (more rods per
  tile) and *stationarity* (smaller tiles) — the spatial correlation length
  of the aberration field is the number that would settle it, and we don't
  have it measured yet.
- Deconvolution then needs a policy: either deconvolve with a *local* PSF
  per tile (spatially-varying deconvolution — more work, more correct), or
  smooth the map to a low-order field model and deconvolve with that.

### Option C — middle ground: a few sub-fields (e.g. 3×3) instead of a full map

- One set per sub-field: catches gross variation, cheap, but the number
  "3×3" is arbitrary and the boundaries are artifacts.
- This is what the week-end region fits already were, done manually.

**My read:** Option B is the right end state (it is also the only option
that *answers* the physical question in §3 by showing the field), and the
cost difference to Option A is small once the planned speed-ups are in.
But **we cannot pick a tile size until we know how fast Z varies**, which
requires a small number of overlapping fits — a one-evening measurement,
not a design decision.

## 5. What happens if we add more Zernike modes (trefoil, spherical)

The user (me) currently fits Z5–Z8. The obvious next modes and their
problems, in order of increasing difficulty:

| mode | name | what it does to the PSF | identifiability problem |
|---|---|---|---|
| Z9, Z10 | trefoil | three-fold asymmetric smearing | needs good orientation coverage of rods (trefoil pattern depends on rod orientation relative to the trefoil axis); weak at our SNR, but in principle identifiable with ~6+ rods at several orientations |
| Z11, Z12, Z13 | spherical aberration (3rd order) | symmetric broadening, more in z than x/y | **strongly degenerate with (a) defocus, (b) the rod's unknown true depth, (c) the capsule width** (we pin width at 0.65 µm — a wrong width is absorbed as spherical-like blur). With rods at unknown, scattered depths, there is no reason the fit could separate "spherical aberration" from "I guessed the width/depth wrong". Also, **spherical aberration from RI mismatch is depth-dependent** (§3), so even if measurable, it would not be one number per tile but a function of z. |
| Z14+ | quadrafoil, higher | very subtle at our magnification/SNR | not worth it before the above work |

Note the asymmetry: **trefoil is a *shape* mode like astigmatism** (it
changes the lateral PSF shape in a way the rod orientation pattern
constrains), so adding Z9/Z10 to the existing fit is a modest, principled
extension. **Spherical aberration is a *width* mode** — it does exactly
what changing the object width does — and our whole identifiability
argument (known-width rods → measure PSF) is the thing that would break.
Fitting spherical aberration would require either beads, or a *measured*
width prior much tighter than 0.5–0.8 µm, or accepting that we can only
fit *defocus-corrected, width-corrected residual* spherical — a weaker
claim. My recommendation: **trefoil: yes, when the 4-mode fit is stable;
spherical: not with rods at unknown depth.**

The map approach changes this slightly: with many small tiles, each tile
has fewer rods, which makes *every* added mode harder (worse conditioning).
So "more modes" and "finer map" pull against each other — the tile size
must be chosen with the *final* mode set in mind, which is another reason
the ordering should be: (1) make 4-mode fits stable, (2) measure the
field, (3) then decide trefoil/spherical.

## 6. The degeneracy that underlies everything (and why defocus was dropped)

We do **not** fit defocus (Z4), tilt (Z2/Z3), or piston (Z1): they are
*exactly degenerate* with things we don't know — each rod's true depth
(absorbs defocus), each rod's lateral position (absorbs tilt). The code
handles this by giving each rod a discrete z-candidate offset (±1.5 µm,
0.5 µm steps) and searching over it. Two consequences:

1. Any *real* defocus of the objective leaks into these per-rod z
   candidates and then partially into the other modes — a small, probably
   tolerable bias, but not zero.
2. The z grid is discrete (0.5 µm ≈ 0.17 waves of defocus). Rods whose true
   offset falls between grid points leave up to ~0.09 waves of unmodeled
   defocus in the residual, which the fit can mis-attribute to astigmatism.
   This is a real, quantifiable source of error in the Z5–Z8 values above,
   and it is why the single-region fits (2 rods) are noisy: with 2 rods the
   z-candidates are nearly the only degrees of freedom.

**This is the same class of problem that would kill spherical-aberration
fitting** (§5): the unknowns we don't model (rod depth, width) absorb the
modes we might want.

## 7. The concrete question for the supervisor

1. **One global set, or a field map?** Given the regional fits in §2, do
   you think the objective/sample system on that microscope can be assumed
   to have a *single* astigmatism/coma over ~200 µm field — or do you
   suspect sample geometry (tilt, RI mismatch, disk alignment) that would
   make it position-dependent? Your prior here decides whether we build the
   map machinery at all.
2. If position-dependent: is the physically expected *pattern* (e.g. linear
   in x from sample tilt, radially symmetric from field astigmatism) known
   for this setup? A parametric model of the *pattern* (fewer parameters
   than a full map) would be a much cheaper middle option.
3. Do you think **spherical aberration is measurable at all** in this
   preparation (rod width uncertainty 0.5–0.8 µm, depth unknown), or should
   we stay at astigmatism/coma (+ trefoil)?
4. For deconvolution purposes, is a *spatially varying* PSF acceptable to
   the analysis pipeline downstream, or do we need to commit to one PSF?
   (DOLPHIN/Huygens both assume one; a per-tile PSF changes the workflow.)

## 8. Field-variation test: workflow and the regions we will use

**How rods get into the fit (current reality):** the automatic rod
extraction (`extract_segments`: Otsu → skeletonize → split → PCA) does not
work on the real data — it merges touching rods, splits bent ones, and
mis-assigns z. The working pipeline is **manual**: I pick one line ROI per
rod in Fiji on the sharpest plane (line endpoints + the z-plane are stored
in the `.roi` file), save the ROI Manager as `RoiSet.zip`, and a script
derives each rod's capsule from the line (xy refined ±3 px, z refined ±3
planes from the axial profile). The fitter only consumes the derived
endpoints `(x0,y0,z0),(x1,y1,z1)`. So *region selection is a manual step*;
a field test must be organized around that.

**Test design (to be run once the current fit finishes):** 3 new regions,
each **8 hand-picked lines at varied orientations, all at a similar z
level** (so the test compares *x,y*, not *z*). Candidate blocks were found
by scanning the whole AF488 channel for 120-px squares containing dense,
z-compact signal (12-plane slabs); 6 blocks exist, 3 chosen for spatial
separation and similar rod z-levels:

| region | field position (µm) | z window (planes / µm) | dense core |
|---|---|---|---|
| **A** | x 98–134, y 16–52 (left-center) | 48–84 / 24–42 | B1, z ≈ 32.5 µm |
| **B** | x 112–147, y 98–134 (center) | 60–96 / 30–48 | B0, z ≈ 39.5 µm |
| **C** | x 180–215, y 30–65 (right) | 96–132 / 48–66 | B2, z ≈ 58.0 µm |

Caveats: (1) A and B share a similar z level (~33–40 µm) — good for a clean
x,y comparison; C sits ~20 µm deeper, which *confounds position with the
depth-dependent aberration from RI mismatch* (§3) — that is partly a
feature (it tests the depth hypothesis directly) but the two effects will
not be separable in one 3-region test. (2) C is near the field edge
(x ≈ 180–215 of 218 µm), where edge effects are expected to be largest.
Cropped TIFs with these windows are on the server
(`/asbdata/moblaende26/zerniky/fieldtest_{A,B,C}.tif`, 36 × 312 × 312);
the 8 line ROIs per crop are to be drawn by hand in Fiji.

**Expected outputs per region:** Z5–Z8 ± bootstrap σ (σ from rod
resampling within the region) + the per-rod z residuals. **Interpretation
rule:** if |Z_i − Z_j| ≳ 2 σ (pooled) for Z5/Z6 between A↔B, the astigmatism
varies in x,y at the ~80–120 µm scale; if not, one global set is
consistent with the data and the region3 pinning is likely a local
(statistical/box) problem, not field variation.

## 9. Current state of the runs (for reference)

- nodeasb01: one fit still running (region3, 7 rods, started 10-30,
  99 %-energy guard band, box 36×393×393 — it will tell us whether the
  guard band fixed the region3 bound-pinning; if it pins again, the
  region-averaging hypothesis of §2 gets stronger).
- The automated "all rods in one box" follow-up fit was cancelled (it would
  have taken ~1 day to answer a question we already suspect has no answer).
- Machine-side facts: fits are CPU-only today; per-measured-rate,
  10–30 Mvox box ≈ 2–6 h per fit. Planned speed-ups (per-rod support
  cropping, parallel tiles, optional GPU FFT batch) are designed but not
  implemented — their *box size* depends on the decision in §7.
