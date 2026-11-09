import pytest
from sibilla.twod import optics

def test_channel_table():
    assert set(optics.CHANNELS) == {"CH1", "CH4"}
    assert optics.CHANNELS["CH1"].lambda_nm == 595.0
    assert optics.CHANNELS["CH1"].kind == "fluorescence"
    assert optics.CHANNELS["CH4"].kind == "transmission"

def test_pupil_config_matches_keyence():
    c = optics.pupil_config("CH1", grid_size=128)
    assert (c.pixel_size, c.na, c.n_im, c.grid_size) == (1.887, 0.20, 1.33, 128)
    assert c.lambda_um == pytest.approx(0.595)

def test_unknown_channel_raises():
    with pytest.raises(KeyError):
        optics.pupil_config("CH3")
