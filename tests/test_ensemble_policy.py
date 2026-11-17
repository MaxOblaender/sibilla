"""Tests for the ensemble combination policy (plans/fix-ensemble-pinned-crops.md).

`combine_ensemble` is the pure decision layer of `fit_ensemble`: given the
per-crop fit results (coefficients + weights), it must

- EXCLUDE any crop whose fit pinned at the coefficient bound (a pinned crop
  carries no direction information — averaging it into good crops corrupts
  them; diagnosed 2026-11-17, `plans/fix-ensemble-pinned-crops.md`),
- keep the declared weights for the surviving crops,
- report what it used and what it rejected,
- and refuse silently-usable output when EVERY crop pinned.
"""

from __future__ import annotations

import pytest

from sibilla.real_pipeline import combine_ensemble

GOOD = {5: 0.08, 6: -0.06, 7: 0.004, 8: -0.002}
PINNED = {5: -0.5, 6: 0.5, 7: -0.476, 8: 0.270}


def _crops(*entries):
    """entries: (coeffs, weight) pairs -> combine_ensemble input format."""
    return [{"name": f"crop{i}", "coeffs": c, "weight": w}
            for i, (c, w) in enumerate(entries)]


class TestCombineEnsemble:
    def test_good_plus_pinned_returns_good(self):
        coeffs, used, rejected = combine_ensemble(
            _crops((GOOD, 4), (PINNED, 4)), coeff_bound=0.5)
        assert coeffs == GOOD
        assert used == ["crop0"]
        assert rejected == ["crop1"]

    def test_weights_respected_among_survivors(self):
        c1 = {**GOOD, 5: 0.06}
        c2 = {**GOOD, 5: 0.12}
        coeffs, used, rejected = combine_ensemble(
            _crops((c1, 2), (c2, 1)), coeff_bound=0.5)
        assert used == ["crop0", "crop1"]
        assert rejected == []
        assert abs(coeffs[5] - (0.06 * 2 + 0.12 * 1) / 3) < 1e-12

    def test_near_bound_is_not_pinned(self):
        near = {**GOOD, 5: 0.5 - 1e-6}       # inside by a hair: admissible
        coeffs, used, _ = combine_ensemble(_crops((near, 1)), coeff_bound=0.5)
        assert used == ["crop0"]
        assert coeffs[5] == pytest.approx(0.5 - 1e-6)

    def test_all_pinned_raises(self):
        with pytest.raises(ValueError, match="pinned"):
            combine_ensemble(_crops((PINNED, 4), (PINNED, 2)), coeff_bound=0.5)

    def test_empty_input_raises(self):
        with pytest.raises(ValueError, match="no crops"):
            combine_ensemble([], coeff_bound=0.5)

    def test_zero_total_weight_raises(self):
        with pytest.raises(ValueError, match="weight"):
            combine_ensemble(_crops((GOOD, 0), (GOOD, 0)), coeff_bound=0.5)

    def test_rejects_pinned_even_when_majority(self):
        good = _crops((PINNED, 9), (GOOD, 1))
        coeffs, used, rejected = combine_ensemble(good, coeff_bound=0.5)
        assert coeffs == GOOD and used == ["crop1"] and rejected == ["crop0"]
