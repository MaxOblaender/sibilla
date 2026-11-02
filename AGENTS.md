# AGENTS.md — conventions for AI agents (and humans) working in this repo

Read this first. It encodes the workflow agreements of the maintainer.

## What this is

SIBILLA (Python) — blind Zernike aberration recovery (Z5–Z8) from 3D fluorescence
stacks, building the aberrated PSF and deconvolving. Python successor to Zoltan
Cseresnyes' C# SIBILLA (`../sibilla_zoltan`, read-only reference — do not modify).

## Git workflow (agreed with maintainer)

- **Default branch: `main`. Never push to `main` unless explicitly told to.**
  `main` must always be green (tests passing).
- **Feature branches from a clean `main`**, named:
  `feat/<topic>`, `fix/<topic>`, `refactor/<topic>`, `perf/<topic>`,
  `docs/<topic>`, `test/<topic>`.
  Always: `git switch main && git pull && git switch -c feat/<topic>`.
  One topic per branch. Long-running experiments must record the commit they
  used (`short-hash`), so results stay reproducible.
- **Commit messages: Conventional Commits** — `feat:`, `fix:`, `refactor:`,
  `perf:`, `test:`, `docs:`, `chore:`, optional scope: `feat(fit): ...`.
  Small, logical commits; message says *why*.
- **Merging is the maintainer's call.** An agent declares a branch ready
  (tests green + working) and waits for the word "merge it". Then:
  `git switch main && git merge --no-ff feat/<topic> && git push && git branch -d ...`
  (`--no-ff` always, so features stay visible/revertible in history).
- **Plans for planned-but-unimplemented work go in `plans/<topic>.md`**,
  committed on the feature branch that will implement them (plan = first commit).
  Do not put plans on `main` unless the maintainer says so.

## Build / test

- Venv: `.venv` (repo-local). Interpreter: `.venv/bin/python`.
  If the venv was moved/renamed, shebangs break — patch with sed on `.venv/bin/*`
  + `pyvenv.cfg`, then `pip install -e . --no-deps`.
- Install: `python3 -m venv .venv && .venv/bin/python -m pip install -e ".[test,data]"`
- **Fast tests** (seconds):
  `.venv/bin/python -m pytest -q tests/test_zernike_basis.py tests/test_psf.py \
    tests/test_psf_confocal.py tests/test_forward.py tests/test_io.py \
    tests/test_real_defaults.py tests/test_segments.py`
- **Full suite** (~20 min, real data present → real fits run):
  `.venv/bin/python -m pytest -q`. Green definition for a branch: all pass.
- Entry points (documented in README): `sibilla.fit_real_crop`,
  `sibilla.deconvolve`, `sibilla.real_pipeline`.

## Hard constraints

- **Never commit data or generated results.** `.gitignore` is a *whitelist*
  (`/*` ignored, explicit `!/` for code+tests+scripts+docs). Never `git add -f`.
  Imaging data (CZI/TIF) lives in the folder but is untracked; on the server it
  is at `/asbdata/moblaende26/zerniky/` (nodeasb01: 256 cores, 2 TB RAM).
- **Never print or commit credentials.** The GitHub token lives in
  `~/.git-credentials` (credential.helper store). Mask it in any output.
- **Throwaway exploration scripts** go to repo root with a `# THROWAWAY:`
  header and are moved to `scripts_extras/` (untracked) when done.
  `scripts/` holds only tracked, still-used workflow front-ends.
- **Physics conventions are pinned** by the test suite (Noll orthonormal basis,
  coefficients in RMS waves, modes Z5–Z8, scalar PSF model, per-rod depth).
  See README "Conventions" and `src/sibilla/defaults.py`. Don't change them
  without the maintainer — the tests encode them deliberately.

## Known state (update this section when it goes stale)

- **Plans live in `plans/`**, tracked file-by-file (see `.gitignore`). Two live
  specs: `manual-structure-selection.md` (manual ROI rod selection — designed,
  not implemented) and `fit-speedups.md` (performance plan — see branch
  `feat/fit-speedups`).
- **Fiji ROI input format (reverse-engineered from a real file):** `RoiSet.zip`
  = one binary `.roi` per ROI (132–156 bytes, `Iout` v0): 4× big-endian float32
  line endpoints at byte 18, int16 z-plane at byte 74, name in the trailing
  bytes. Parser example: `scripts_extras/rod_probe.py`. z IS stored.
- **Current fit-quality issues (open):** cluster fits can be unstable —
  region3 pinned at the ±0.5-wave bound, suspected cause: cyclic-FFT box guard
  band too small (2 µm fixed vs ~7 µm needed). The 99%-energy margin fix
  exists in `scripts_extras/rod_fit*` / server runs but is NOT yet in the
  package. Weekend fit logs: nodeasb01 `/asbdata/moblaende26/zerniky/weekend_logs/`.
- The fitter is single-threaded and slow on big boxes; performance options are
  documented in `plans/fit-speedups.md`.

## Handoff to a new agent (maintainer's standard prompt)

> Work in `~/Documents/git_repos/sibilla`. Read `AGENTS.md` first. Branch off
> `main` with `feat/<topic>`, Conventional Commits, don't touch `main`.
