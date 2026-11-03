"""Tests of the compiled SAM 3.1 mask decoder against Meta's."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from meta_sam_parser import (
    InvalidSegmentationMaskError, SegmentationMaskRecord, decode_mask_to_raster,
)

from pivot.meta import mask_codec
from pivot.meta.sam import _parse, parse_sam_output

DATA = Path(__file__).parent / "data"


def _masks(name: str) -> list:
    result = _parse((DATA / name).read_text(encoding="utf-8"), "video" if "video" in name
                    else "image")
    return [r.mask for r in result.records if isinstance(r, SegmentationMaskRecord)]


@pytest.mark.parametrize("name", ["sam_video_chair.txt", "sam_image_chairs.txt"])
def test_masks_are_decoded_as_by_meta(name: str) -> None:
    masks = _masks(name)
    assert masks
    for mask in masks:
        expected = np.frombuffer(decode_mask_to_raster(mask), dtype=np.uint8)
        assert np.array_equal(mask_codec.decode_mask(mask),
                              expected.reshape(mask.height, mask.width).astype(bool))


def test_parsed_detections_keep_the_masks_checked_by_the_parser() -> None:
    text = (DATA / "sam_video_chair.txt").read_text(encoding="utf-8")
    detections = parse_sam_output(text, "video")
    for detection, mask in zip(detections, _masks("sam_video_chair.txt")):
        expected = np.frombuffer(decode_mask_to_raster(mask), dtype=np.uint8)
        assert np.array_equal(detection.mask,
                              expected.reshape(mask.height, mask.width).astype(bool))


def test_unpacking_rejects_what_is_not_canonical() -> None:
    payload = _masks("sam_video_chair.txt")[0].payload[1:]
    assert mask_codec.unpack(payload) is not None
    assert mask_codec.unpack(payload[:-1]) is None
    assert mask_codec.unpack(payload[:-1] + '"') is None
    assert mask_codec.unpack("") is None


@pytest.mark.parametrize("cut", [slice(None, -3), slice(None, 30)])
def test_a_broken_mask_raises_meta_s_error(cut: slice) -> None:
    mask = _masks("sam_video_chair.txt")[0]
    with pytest.raises(InvalidSegmentationMaskError):
        mask_codec.decode_mask(replace(mask, payload=mask.payload[cut]))
