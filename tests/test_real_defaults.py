"""Tests for real-data defaults (Task 1)."""

from __future__ import annotations

import pytest

from sibilla.defaults import (
    BACTERIA_SCENE,
    REAL_ANCHOR_BAND_UM,
    REAL_ANCHOR_UM,
    REAL_CHANNEL,
    REAL_CHANNEL_LAMBDA_NM,
    REAL_CZI_NAME,
    REAL_CROP_NAME,
    REAL_LAMBDA_NM,
    REAL_MODES,
    REAL_NA,
    REAL_N_IM,
    REAL_PIXEL_UM,
    REAL_Z_STEP_UM,
    ROI_RADIUS_UM,
    Z_CANDIDATE_GRID_UM,
    real_pupil_config,
)


def test_bacteria_scene_fields():
    assert BACTERIA_SCENE.fov_px == 256
    assert BACTERIA_SCENE.pixel_size_um == pytest.approx(0.1135)
    assert BACTERIA_SCENE.n_planes == 15
    assert BACTERIA_SCENE.z_step_um == pytest.approx(0.5)
    assert BACTERIA_SCENE.n_segments == 10
    assert BACTERIA_SCENE.length_range_um == (1.0, 3.0)
    assert BACTERIA_SCENE.min_sep_um == pytest.approx(1.5)
    assert BACTERIA_SCENE.border_margin_um == pytest.approx(1.5)
    assert BACTERIA_SCENE.max_angular_gap_deg == 100
    assert BACTERIA_SCENE.fluorescent_diameter_um == pytest.approx(0.65)
    assert BACTERIA_SCENE.supersample == 8


def test_z_candidate_grid():
    assert Z_CANDIDATE_GRID_UM[0] == pytest.approx(-1.5)
    assert Z_CANDIDATE_GRID_UM[-1] == pytest.approx(1.5)
    steps = [
        Z_CANDIDATE_GRID_UM[i + 1] - Z_CANDIDATE_GRID_UM[i]
        for i in range(len(Z_CANDIDATE_GRID_UM) - 1)
    ]
    assert all(s == pytest.approx(0.5) for s in steps)


def test_real_constants():
    assert REAL_PIXEL_UM == pytest.approx(0.1135)
    assert REAL_Z_STEP_UM == pytest.approx(0.5)
    assert REAL_LAMBDA_NM == pytest.approx(517.0)
    assert REAL_NA == pytest.approx(1.2)
    assert REAL_N_IM == pytest.approx(1.33)
    assert REAL_ANCHOR_UM == pytest.approx(0.65)
    assert REAL_ANCHOR_BAND_UM == (0.5, 0.8)
    assert REAL_MODES == (5, 6, 7, 8)
    assert REAL_CHANNEL == 1
    assert REAL_CHANNEL_LAMBDA_NM[1] == pytest.approx(517.0)
    assert REAL_CHANNEL_LAMBDA_NM[0] == pytest.approx(465.0)
    assert REAL_CHANNEL_LAMBDA_NM[2] == pytest.approx(568.0)
    assert REAL_CHANNEL_LAMBDA_NM[4] == pytest.approx(636.0)
    assert REAL_CZI_NAME == "GoC-Exp4_C3_7.czi"
    assert REAL_CROP_NAME == "GoC-Exp4_C3_7-1_3d.tif"
    assert ROI_RADIUS_UM == pytest.approx(1.5)


def test_real_pupil_config():
    cfg = real_pupil_config(256)
    assert cfg.na == pytest.approx(1.2)
    assert cfg.lambda_um == pytest.approx(0.517)
    assert cfg.n_im == pytest.approx(1.33)
    assert cfg.grid_size == 256
    assert cfg.pixel_size == pytest.approx(0.1135)
    # per-channel override
    cfg2 = real_pupil_config(256, lambda_nm=568.0)
    assert cfg2.lambda_um == pytest.approx(0.568)
