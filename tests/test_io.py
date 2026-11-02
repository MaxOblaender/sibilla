"""Tests for io.py (Tasks 2–3)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("czifile")

from sibilla.io import (  # noqa: E402
    ChannelInfo,
    RealMeta,
    _parse_metadata_xml,
    cut_crop_3d,
    locate_stack,
    locate_z_band,
    load_channel,
    read_czi_metadata,
    read_crop_tiff,
    square_pad,
)

REPO = Path(__file__).resolve().parents[2]
CZI = REPO / "GoC-Exp4_C3_7.czi"
CROP = REPO / "GoC-Exp4_C3_7-1_3d.tif"

needs_czi = pytest.mark.skipif(not CZI.exists(), reason="CZI not present")
needs_crop = pytest.mark.skipif(not CROP.exists(), reason="3D crop not present")


_XML = """<?xml version="1.0"?>
<ImageDocument>
 <Metadata>
  <Information>
   <Image>
    <Dimensions>
     <Channels>
      <Channel Id="Channel:0" Name="DAPI"><DyeName>DAPI</DyeName>
       <ExcitationWavelength>353</ExcitationWavelength>
       <EmissionWavelength>465</EmissionWavelength></Channel>
      <Channel Id="Channel:1" Name="AF488"><DyeName>Alexa Fluor 488</DyeName>
       <ExcitationWavelength>493</ExcitationWavelength>
       <EmissionWavelength>517</EmissionWavelength></Channel>
      <Channel Id="Channel:3" Name="Bright"><DyeName>TL Brightfield</DyeName></Channel>
     </Channels>
     <SizeX>1920</SizeX><SizeY>1440</SizeY><SizeZ>196</SizeZ><SizeC>6</SizeC>
    </Dimensions>
    <ObjectiveSettings><ObjectiveRef Id="Objective:1"/></ObjectiveSettings>
   </Image>
   <Objectives>
    <Objective Id="Objective:1" Name="LD LCI Plan-Apochromat 40x/1.2 Imm autocorr DIC">
     <LensNA>1.2</LensNA><NominalMagnification>40</NominalMagnification>
     <ImmersionRefractiveIndex>1.33</ImmersionRefractiveIndex>
     <Immersion>Water</Immersion></Objective>
   </Objectives>
  </Information>
  <Scaling>
   <Items>
    <Distance Id="X"><Value>1.135E-07</Value></Distance>
    <Distance Id="Y"><Value>1.135E-07</Value></Distance>
    <Distance Id="Z"><Value>5E-07</Value></Distance>
   </Items>
  </Scaling>
  <DisplaySetting/>
 </Metadata>
</ImageDocument>
"""

class TestParseMetadataXml:
    def test_voxel_and_shape(self):
        meta = _parse_metadata_xml(_XML)
        # XML values are in metres; RealMeta exposes µm
        assert meta.voxel_um == pytest.approx((0.1135, 0.1135, 0.5))
        assert meta.shape == (6, 196, 1440, 1920)
        assert meta.na == pytest.approx(1.2)
        assert meta.n_im == pytest.approx(1.33)
        assert meta.magnification == 40
        assert meta.objective_name.startswith("LD LCI Plan-Apochromat")

    def test_channels(self):
        meta = _parse_metadata_xml(_XML)
        by_name = {c.name: c for c in meta.channels}
        assert by_name["AF488"].index == 1
        assert by_name["AF488"].emission_nm == pytest.approx(517.0)
        assert by_name["AF488"].excitation_nm == pytest.approx(493.0)
        assert by_name["AF488"].dye == "Alexa Fluor 488"
        assert by_name["DAPI"].emission_nm == pytest.approx(465.0)
        assert by_name["Bright"].emission_nm is None

    def test_missing_objective_reference_raises(self):
        bad = _XML.replace('<ObjectiveRef Id="Objective:1"/>',
                           '<ObjectiveRef Id="Objective:9"/>')
        with pytest.raises(ValueError, match="objective"):
            _parse_metadata_xml(bad)


@needs_czi
class TestReadCziMetadata:
    def test_real_meta(self):
        meta = read_czi_metadata(CZI)
        assert isinstance(meta, RealMeta)
        assert meta.voxel_um == pytest.approx((0.1135, 0.1135, 0.5))
        assert meta.shape == (6, 196, 1440, 1920)
        assert meta.na == pytest.approx(1.2)
        assert meta.n_im == pytest.approx(1.33)
        assert meta.magnification == 40
        assert len(meta.channels) == 6
        af = [c for c in meta.channels if c.name == "AF488"]
        assert len(af) == 1 and af[0].index == 1
        assert af[0].emission_nm == pytest.approx(517.0)


# --- Task 3: channel load, crop cut/locate, square pad ------------------------


def _synthetic_volume(seed: int = 0, nz: int = 8, m: int = 60, n: int = 80,
                      blob_at: tuple[int, int] = (30, 40)) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vol = rng.poisson(5.0, size=(nz, m, n)).astype(np.float32)
    yy, xx = np.mgrid[0:m, 0:n]
    by, bx = blob_at
    blob = np.exp(-((yy - by) ** 2 + (xx - bx) ** 2) / 18.0)
    for z in range(nz):
        vol[z] += 400.0 * blob
    return vol


class TestSquarePad:
    def test_pads_to_square_with_per_plane_median(self):
        rng = np.random.default_rng(1)
        stack = rng.poisson(50, size=(15, 184, 256)).astype(np.float32)
        out = square_pad(stack)
        assert out.shape == (15, 256, 256)
        np.testing.assert_allclose(out[:, :184, :256], stack)
        for k in range(15):
            med = np.median(stack[k])
            assert np.all(out[k, 184:, :] == med)
            assert np.all(out[k, :, 256:] == med)

    def test_already_square_is_copy(self):
        stack = np.zeros((3, 8, 8), np.float32)
        out = square_pad(stack)
        assert out.shape == (3, 8, 8)
        assert out is not stack


class TestLocateStack:
    def test_finds_blob(self):
        vol = _synthetic_volume()
        ncc, y0, x0 = locate_stack(vol, vol[4])
        assert ncc > 0.99
        assert (y0, x0) == (0, 0)

    def test_finds_shifted_template(self):
        vol = _synthetic_volume(blob_at=(30, 40))
        tmpl = vol[4][10:50, 20:60]          # window containing the blob
        ncc, y0, x0 = locate_stack(vol, tmpl)
        assert ncc > 0.95
        assert (y0, x0) == (10, 20)

    def test_wrong_location_rejected(self):
        # template of pure noise (no blob) must not "match" well anywhere
        rng = np.random.default_rng(2)
        noise_tmpl = rng.poisson(5.0, size=(20, 20)).astype(np.float32)
        vol = _synthetic_volume()
        ncc, _, _ = locate_stack(vol, noise_tmpl)
        assert ncc < 0.8


@needs_czi
class TestRealChannel:
    def test_load_channel_shape(self):
        vol = load_channel(CZI, 1)
        assert vol.shape == (196, 1440, 1920)
        assert vol.dtype == np.float32

    def test_cut_crop_matches_delivered_tiff(self):
        cut = cut_crop_3d(CZI, 1, 7, 21, 1100, 116, 184, 256)
        stack, tags = read_crop_tiff(CROP)
        assert cut.shape == stack.shape == (15, 184, 256)
        np.testing.assert_allclose(cut, stack, atol=5)
        assert tags.get("pixel_size_um") == pytest.approx(0.1135, rel=1e-3)
        assert tags.get("z_step_um") == pytest.approx(0.5)

    def test_locate_z_band_finds_delivered_crop(self):
        vol = load_channel(CZI, 1)
        stack, _ = read_crop_tiff(CROP)
        ncc, z0 = locate_z_band(vol, stack, 1100, 116)
        assert z0 == 7
        assert ncc > 0.99
