# arael Python support library, vendored into every generated
# package: math value types (doubling as the FFI structs), the pinhole
# camera (cameraf / camerad), the g2o reader, the shared solver
# surface, and the column transport behind the bulk collection calls.

from . import columns, g2o, geometry, math, solver, transform
from .geometry import Camera, camerad, cameraf
from .math import (matrix2d, matrix2f, matrix3d, matrix3f, quaternd,
                   quaternf, vect2d, vect2f, vect3d, vect3f)
from .solver import AraelError, CovMode, CovOrdering, LmPreset, LmStatus, LmTiming
from .transform import (scaled_transform3d, scaled_transform3f, transform3d,
                        transform3f)

__all__ = [
    "columns", "g2o", "geometry", "math", "solver", "transform", "Camera", "cameraf", "camerad",
    "vect2f", "vect2d", "vect3f", "vect3d",
    "matrix2f", "matrix2d", "matrix3f", "matrix3d",
    "quaternf", "quaternd",
    "transform3d", "transform3f", "scaled_transform3d", "scaled_transform3f",
    "AraelError", "CovMode", "CovOrdering", "LmPreset", "LmStatus", "LmTiming",
]
