"""Tests for run_preflight.py (Task 8, simplified: P1, P2, P5, P6)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sibilla.run_preflight import (
    g2_verdict,
    preflight_experiments,
    run_preflight,
)


class TestRegistry:
    def test_four_experiments_registered(self):
        exps = {e.name: e for e in preflight_experiments()}
        for name in ("P1_null", "P2_combined_depth", "P5_segment_count",
                     "P6_rl_roundtrip"):
            assert name in exps, name
        # the sensitivity studies (old P3/P4/P7) are dropped by decision
        assert "P3_anchor" not in exps
        assert "P4_lambda" not in exps

    def test_pinned_values(self):
        exps = {e.name: e for e in preflight_experiments()}
        assert exps["P1_null"].truth == {}
        assert exps["P2_combined_depth"].truth == {
            5: 0.10, 6: 0.10, 7: 0.07, 8: 0.07}
        assert exps["P2_combined_depth"].peak == pytest.approx(4500.0)
        assert exps["P2_combined_depth"].z_scatter_um == pytest.approx(1.0)
        assert exps["P5_segment_count"].variants == (1, 3, 6, 10, 20)


class TestG2Verdict:
    def test_pass(self):
        results = {"P2_combined_depth": {"per_mode_err": {5: 0.01, 6: 0.02,
                                                          7: 0.03, 8: 0.04}}}
        v = g2_verdict(results)
        assert v["pass"] is True

    def test_fail_on_006(self):
        results = {"P2_combined_depth": {"per_mode_err": {5: 0.01, 6: 0.06,
                                                          7: 0.01, 8: 0.01}}}
        v = g2_verdict(results)
        assert v["pass"] is False


class TestSmoke:
    def test_p1_null_smoke(self, tmp_path: Path, monkeypatch):
        """P1_null on a reduced scene completes and writes result.json."""
        from dataclasses import replace

        from sibilla import run_preflight as rp

        small = rp._capsule_spec(96, n_planes=11)
        orig = rp.preflight_experiments

        def reduced():
            out = []
            for e in orig():
                if e.name == "P1_null":
                    e = replace(e, spec=small)
                out.append(e)
            return out

        monkeypatch.setattr(rp, "preflight_experiments", reduced)
        results = run_preflight(tmp_path, only=["P1_null"])
        assert "P1_null" in results
        rec = json.loads((tmp_path / "P1_null" / "result.json").read_text())
        assert "per_mode_err" in rec
