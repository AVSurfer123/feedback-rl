import typing

from .spline import Spline
from .bspline_path import BSpline
from .const_accel_path import ConstAccelSpline

SPLINE_MAP: typing.Dict[int, Spline] = {
    0: ConstAccelSpline,
    1: BSpline
}
