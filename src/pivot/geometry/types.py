"""Shared array type aliases for the geometry package."""

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]
IntArray = npt.NDArray[np.int64]
UInt8Array = npt.NDArray[np.uint8]
