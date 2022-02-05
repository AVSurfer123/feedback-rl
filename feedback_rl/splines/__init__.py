from typing import Dict
from .spline import Spline
from .bspline_path import BSpline
from .const_accel_path import ConstAccelSpline

SPLINE_MAP: Dict[int, Spline] = {
    0: ConstAccelSpline,
    1: BSpline
}
