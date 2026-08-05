# arael Python support library, vendored into every generated
# package: math value types (doubling as the FFI structs), the pinhole
# camera (cameraf / camerad), the g2o reader, and the shared solver
# surface.

from . import g2o, geometry, math, solver
from .geometry import Camera, camerad, cameraf
from .math import (matrix2d, matrix2f, matrix3d, matrix3f, quaternd,
                   quaternf, vect2d, vect2f, vect3d, vect3f)
from .solver import AraelError, CovMode, CovOrdering, LmPreset, LmStatus, LmTiming

__all__ = [
    "g2o", "geometry", "math", "solver", "Camera", "cameraf", "camerad",
    "vect2f", "vect2d", "vect3f", "vect3d",
    "matrix2f", "matrix2d", "matrix3f", "matrix3d",
    "quaternf", "quaternd",
    "AraelError", "CovMode", "CovOrdering", "LmPreset", "LmStatus", "LmTiming",
]
