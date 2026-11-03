"""Decoding SAM 3.1 masks with compiled code, when numba is installed.

Meta's ``meta-sam-parser`` decodes each mask pixel by pixel in pure Python:
an adaptive range coder over prediction residuals, about 0.15 s per kilobyte
of payload, which is most of the time spent on a video's responses. This is
the same decoder for the ``lossless`` encoding (the one SAM 3.1 sends),
compiled by numba, giving the same pixels. Meta's parser also decodes every
mask once to check it, and drops the pixels: :func:`install` has it check
them with this decoder, and :func:`keep_decoded` keeps them, so that each mask
is decoded once.

Anything unusual (another encoding, a malformed payload, numba missing) goes
to Meta's decoder, which raises the errors and makes the checks it makes.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import numpy as np
from meta_sam_parser import SegmentationMask, decode_mask_to_raster
from meta_sam_parser import _segmentation

from pivot.geometry.types import BoolArray, UInt8Array

try:
    from numba import njit
except ImportError:  # pragma: no cover - depends on the environment
    njit = None

# Characters of the payload's base-85 alphabet, in order (as in meta-sam-parser).
_EXCLUDED = frozenset('"\\,;<>|')
_ALPHABET = "".join(chr(code) for code in range(0x21, 0x7F) if chr(code) not in _EXCLUDED)
_RADIX = 85
_PREFIX_LENGTH = 5
_DIGITS = np.full(128, -1, dtype=np.int64)
_DIGITS[[ord(character) for character in _ALPHABET]] = np.arange(len(_ALPHABET))

# Model constants of the lossless coder (as in meta-sam-parser).
_SYMBOLS = 256
_ZERO_INCREMENT = 32
_ZERO_LIMIT = 16384
_ORDER_ONE_INCREMENT = 56
_ORDER_ONE_LIMIT = 8192
_ORDER_TWO_STEP = 14 * 44
_ORDER_TWO_LIMIT = 14 * 3584
_ORDER_ZERO_STEP = 2 * 16
_ORDER_ZERO_LIMIT = 2 * 2048
_ORDER_ZERO_INITIAL = 2
_ORDER_ONE_INITIAL = 1
_TOP = 1 << 24
_BOTTOM = 1 << 16
_MASK_32 = 0xFFFFFFFF
_MASK_16 = 0xFFFF
_SPATIAL_MODES = 5


def unpack(payload: str) -> UInt8Array | None:
    """Turns a base-85 payload (without its leading ``~``) into bytes.

    Args:
        payload: The packed payload: a 5-digit length, then 5 digits per 4 bytes.

    Returns:
        The bytes, as uint8; None if the payload is not well formed.
    """
    codes = np.frombuffer(payload.encode("ascii", errors="replace"), dtype=np.uint8)
    if len(codes) < _PREFIX_LENGTH or len(codes) != len(payload):
        return None
    digits = _DIGITS[np.minimum(codes, 127)]
    if (digits < 0).any():
        return None
    length = 0
    for digit in digits[:_PREFIX_LENGTH]:
        length = length * _RADIX + int(digit)
    full, remainder = divmod(length, 4)
    expected = _PREFIX_LENGTH + full * 5 + (remainder + 1 if remainder else 0)
    if len(codes) != expected:
        return None
    body = digits[_PREFIX_LENGTH:]
    if remainder:
        # The tail is padded with the highest digit, as the encoder does.
        body = np.concatenate([body, np.full(4 - remainder, _RADIX - 1, dtype=np.int64)])
    values = body.reshape(-1, 5) @ (_RADIX ** np.arange(4, -1, -1, dtype=np.int64))
    if (values > _MASK_32).any():
        return None
    shifts = np.array([24, 16, 8, 0], dtype=np.int64)
    out: UInt8Array = ((values[:, None] >> shifts) & 0xFF).astype(np.uint8).ravel()[:length]
    if remainder:
        # Canonical tail: its digits are those of its bytes padded with zeros.
        value = int.from_bytes(bytes(out[full * 4:]) + bytes(4 - remainder), "big")
        tail = [(value // _RADIX ** power) % _RADIX for power in range(4, -1, -1)]
        if tail[:remainder + 1] != [int(digit) for digit in digits[-(remainder + 1):]]:
            return None
    return out


def _decode_lossless(packed: UInt8Array, width: int, height: int,
                     output: UInt8Array) -> int:  # pragma: no cover - compiled
    """Decodes a lossless raster, as ``_decode_lossless_raster`` in meta-sam-parser.

    Args:
        packed: The unpacked payload: a predictor byte, then the range-coded data.
        width: Raster width.
        height: Raster height.
        output: (width * height,) uint8, filled with the coverage values.

    Returns:
        0 on success; 1 if the payload is malformed.
    """
    # pylint: disable=too-many-locals,too-many-branches,too-many-statements
    # pylint: disable=too-many-return-statements
    selector = int(packed[0])
    data = packed[1:]
    size = len(data)
    if size < 4:
        return 1
    position = 0
    low = 0
    rng = _MASK_32
    code = 0
    for _ in range(4):
        code = ((code << 8) | int(data[position])) & _MASK_32
        position += 1

    length = width * height
    residuals = np.zeros(length, dtype=np.int64)
    zero_counts = np.ones(32, dtype=np.int64)
    order_one = np.full(_SYMBOLS * _SYMBOLS, _ORDER_ONE_INITIAL, dtype=np.int64)
    order_one_totals = np.full(_SYMBOLS, _SYMBOLS * _ORDER_ONE_INITIAL, dtype=np.int64)
    order_zero = np.full(_SYMBOLS, _ORDER_ZERO_INITIAL, dtype=np.int64)
    order_zero_total = _SYMBOLS * _ORDER_ZERO_INITIAL
    # Order-two tables are made when first used: a slot per context, in a pool that grows.
    slots = np.full(_SYMBOLS * _SYMBOLS, -1, dtype=np.int64)
    pool = np.zeros((64, _SYMBOLS + 1), dtype=np.int64)
    used = 0
    previous = 0

    for offset in range(length):
        x = offset % width
        y = offset // width
        left_zero = (residuals[offset - 1] == 0) if x > 0 else True
        up_zero = (residuals[offset - width] == 0) if y > 0 else True
        ul_zero = (residuals[offset - width - 1] == 0) if x > 0 and y > 0 else True
        ur_zero = (residuals[offset - width + 1] == 0) if y > 0 and x < width - 1 else True
        zero_offset = (int(left_zero) | (int(up_zero) << 1) | (int(ul_zero) << 2)
                       | (int(ur_zero) << 3)) << 1
        zero = zero_counts[zero_offset]
        nonzero = zero_counts[zero_offset + 1]

        # Range decoder: frequency, then decode (see _RangeDecoder).
        scaled = rng // (zero + nonzero)
        if scaled == 0:
            return 1
        value = ((code - low) & _MASK_32) // scaled
        if value >= zero + nonzero:
            value = zero + nonzero - 1
        bit = 0 if value < zero else 1
        cumulative = 0 if bit == 0 else zero
        frequency = zero if bit == 0 else nonzero
        low = (low + scaled * cumulative) & _MASK_32
        rng = scaled * frequency
        while True:
            if ((low ^ ((low + rng) & _MASK_32)) & _MASK_32) < _TOP:
                pass
            elif rng < _BOTTOM:
                rng = ((-low) & _MASK_32) & (_BOTTOM - 1)
            else:
                break
            if position >= size:
                return 1
            code = ((code << 8) | int(data[position])) & _MASK_32
            position += 1
            low = (low << 8) & _MASK_32
            rng = (rng << 8) & _MASK_32

        zero_counts[zero_offset + bit] = (zero_counts[zero_offset + bit]
                                          + _ZERO_INCREMENT) & _MASK_32
        if zero_counts[zero_offset] + zero_counts[zero_offset + 1] >= _ZERO_LIMIT:
            zero_counts[zero_offset] = max(zero_counts[zero_offset] >> 1, 1)
            zero_counts[zero_offset + 1] = max(zero_counts[zero_offset + 1] >> 1, 1)

        symbol = 0
        if bit == 1:
            above = residuals[offset - width] if offset >= width else 0
            key = (previous << 8) | above
            slot = slots[key]
            if slot < 0:
                if used == pool.shape[0]:
                    grown = np.zeros((2 * used, _SYMBOLS + 1), dtype=np.int64)
                    grown[:used] = pool
                    pool = grown
                slot = used
                slots[key] = slot
                used += 1
            second = pool[slot]
            first = previous << 8
            excluded = second[0] + order_one[first] + order_zero[0]
            total = second[_SYMBOLS] + order_one_totals[previous] + order_zero_total - excluded
            scaled = rng // total
            if scaled == 0:
                return 1
            target = ((code - low) & _MASK_32) // scaled
            if target >= total:
                target = total - 1
            cumulative = 0
            symbol = 1
            frequency = second[1] + order_one[first + 1] + order_zero[1]
            while cumulative + frequency <= target:
                cumulative += frequency
                symbol += 1
                if symbol >= _SYMBOLS:
                    return 1
                frequency = second[symbol] + order_one[first + symbol] + order_zero[symbol]
            low = (low + scaled * cumulative) & _MASK_32
            rng = scaled * frequency
            while True:
                if ((low ^ ((low + rng) & _MASK_32)) & _MASK_32) < _TOP:
                    pass
                elif rng < _BOTTOM:
                    rng = ((-low) & _MASK_32) & (_BOTTOM - 1)
                else:
                    break
                if position >= size:
                    return 1
                code = ((code << 8) | int(data[position])) & _MASK_32
                position += 1
                low = (low << 8) & _MASK_32
                rng = (rng << 8) & _MASK_32

            second[symbol] = (second[symbol] + _ORDER_TWO_STEP) & _MASK_16
            second[_SYMBOLS] = (second[_SYMBOLS] + _ORDER_TWO_STEP) & _MASK_16
            if second[_SYMBOLS] >= _ORDER_TWO_LIMIT:
                rescaled = 0
                for index in range(_SYMBOLS):
                    second[index] = second[index] >> 1
                    rescaled += second[index]
                second[_SYMBOLS] = rescaled & _MASK_16
            order_one[first + symbol] = (order_one[first + symbol]
                                         + _ORDER_ONE_INCREMENT) & _MASK_16
            order_one_totals[previous] += _ORDER_ONE_INCREMENT
            if order_one_totals[previous] >= _ORDER_ONE_LIMIT:
                rescaled = 0
                for index in range(_SYMBOLS):
                    halved = max(order_one[first + index] >> 1, 1)
                    order_one[first + index] = halved
                    rescaled += halved
                order_one_totals[previous] = rescaled
            order_zero[symbol] = (order_zero[symbol] + _ORDER_ZERO_STEP) & _MASK_16
            order_zero_total += _ORDER_ZERO_STEP
            if order_zero_total >= _ORDER_ZERO_LIMIT:
                rescaled = 0
                for index in range(_SYMBOLS):
                    order_zero[index] = order_zero[index] >> 1
                    rescaled += order_zero[index]
                order_zero_total = rescaled

        residuals[offset] = symbol
        # Spatial prediction of the coverage byte (see _predict_byte).
        left = int(output[offset - 1]) if x > 0 else 0
        up = int(output[offset - width]) if y > 0 else 0
        if selector == 1:
            prediction = up
        elif selector == 2:
            prediction = left
        elif selector == 3:
            prediction = (left + up) >> 1
        else:
            upper_left = int(output[offset - width - 1]) if x > 0 and y > 0 \
                else 0
            if selector == 0:
                estimate = left + up - upper_left
                left_distance = abs(estimate - left)
                up_distance = abs(estimate - up)
                ul_distance = abs(estimate - upper_left)
                if left_distance <= up_distance and left_distance <= ul_distance:
                    prediction = left
                elif up_distance <= ul_distance:
                    prediction = up
                else:
                    prediction = upper_left
            elif upper_left >= max(left, up):
                prediction = min(left, up)
            elif upper_left <= min(left, up):
                prediction = max(left, up)
            else:
                prediction = left + up - upper_left
        delta = -((symbol + 1) >> 1) if symbol & 1 else symbol >> 1
        output[offset] = (prediction + delta) & 0xFF
        previous = symbol
    return 0


_compiled: Any = njit(cache=True, nogil=True)(_decode_lossless) if njit is not None else None


def compiled() -> bool:
    """Says whether masks are decoded by compiled code.

    Returns:
        True when numba is installed.
    """
    return _compiled is not None


def decode_mask(mask: SegmentationMask) -> BoolArray:
    """Decodes a SAM 3.1 mask into its pixels.

    Args:
        mask: The mask of a record from Meta's parser.

    Returns:
        (height, width) bool pixels.
    """
    width, height = mask.width, mask.height
    usual = mask.encoding == "lossless" and mask.payload.startswith("~")
    if _compiled is not None and usual and width > 0 and height > 0:
        packed = unpack(mask.payload[1:])
        if packed is not None and len(packed) >= 5 and packed[0] < _SPATIAL_MODES:
            coverage = np.empty(width * height, dtype=np.uint8)
            if _compiled(packed, width, height, coverage) == 0:
                return (coverage >= 129).reshape(height, width)
    # Meta's decoder raises the right error, or decodes what this one does not.
    raster = np.frombuffer(decode_mask_to_raster(mask), dtype=np.uint8)
    return raster.reshape(mask.height, mask.width).astype(bool)


# Masks decoded while checked by the parser, by id, in the thread parsing them.
_local = threading.local()


def _check(mask: SegmentationMask) -> bytes:
    """Stands for ``decode_mask_to_raster`` in Meta's parser, which calls it to check a mask.

    Args:
        mask: The mask to check.

    Returns:
        Its raster, one byte per pixel.
    """
    pixels = decode_mask(mask)
    kept: dict[int, BoolArray] | None = getattr(_local, "kept", None)
    if kept is not None:
        kept[id(mask)] = pixels
    return pixels.astype(np.uint8).tobytes()


def install() -> None:
    """Has Meta's parser check masks with the compiled decoder (nothing without numba)."""
    if _compiled is not None:
        setattr(_segmentation, "decode_mask_to_raster", _check)


@contextmanager
def keep_decoded() -> Iterator[dict[int, BoolArray]]:
    """Keeps the pixels of the masks the parser checks in this thread, by mask id.

    Yields:
        The pixels, by ``id(mask)``: valid while the masks are alive.
    """
    _local.kept = {}
    try:
        yield _local.kept
    finally:
        _local.kept = None
