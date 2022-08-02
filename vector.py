import math
from typing import Sequence

import numpy as np

# Shorthand variables
PI = math.pi
TAU = math.tau
TWO_PI = TAU
RIGHT_ANGLE = PI / 2


def radians(degrees: float) -> float:
    """Convert degrees to radians"""
    return degrees * PI / 180.0


def degrees(radians: float) -> float:
    """Convert radians to degrees"""
    return radians * 180.0 / PI


class Vector(np.ndarray):
    """A vector class that extends numpy.ndarray with some useful methods."""

    def __new__(cls, input_array: Sequence | None = None):
        """Create a new vector. Raise an error if the input is not one dimensional."""
        arr = np.array(input_array, dtype=np.float64)
        if arr.ndim != 1:
            raise ValueError("Vector must be 1-dimensional")
        return np.array(input_array, dtype=np.float64).view(cls)

    def __array_ufunc__(self, ufunc, method, *args, out=None, **kwargs):
        """Try to keep the default behavior of numpy.ndarray for operations."""
        args = [x.view(np.ndarray) if isinstance(x, Vector) else x for x in args]
        if out:
            outs = [x.view(np.ndarray) if isinstance(x, Vector) else x for x in out]
            kwargs["out"] = tuple(outs)
        else:
            outs = (None,) * ufunc.nout

        results = super().__array_ufunc__(ufunc, method, *args, **kwargs)
        if results is NotImplemented:
            return NotImplemented

        if ufunc.nout == 1:
            results = (results,)

        results = tuple(
            (np.asarray(result).view(Vector) if out is None else out)
            for result, out in zip(results, outs)
        )

        return results[0] if len(results) == 1 else results

    def magnitude(self) -> float:
        """Get the magnitude of a vector"""
        return np.linalg.norm(self)

    def normalize(self) -> "Vector":
        """Normalize the vector v - Get the unit vector of v"""
        return self / self.magnitude(self)

    def angle(self, other=None, radians: bool = True) -> float:
        """Get the angle between this vector and another, or between this vector and the x-axis. Max angle is PI (180 degrees)"""
        if other is None:
            other = np.zeros(self.shape).view(Vector)
            other[0] = 1.0
        if not isinstance(other, Vector):
            other = Vector(other)
        result = np.arccos(self.dot(other) / (self.magnitude() * other.magnitude()))
        # result = np.arccos(np.dot(self, other) / (self.magnitude() * v2.magnitude()))
        return result if radians else degrees(result)

    def set_magnitude(self, magnitude: float) -> "Vector":
        """Set the magnitude of the vector"""
        return self * (magnitude / self.magnitude())

    def set_angle(self, angle: float, radians: bool = True) -> "Vector":
        """Set the angle of the vector"""  # TODO
        return NotImplemented
