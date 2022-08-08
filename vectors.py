import numpy as np


def normalize_angle(angle: float, radians: bool = True) -> float:
    """Wrap the heading to be -PI < heading <= PI"""
    rad_angle = angle if radians else np.deg2rad(angle)

    if -np.pi < rad_angle <= np.pi:
        return angle

    result = abs(rad_angle) % np.pi  # modulo only works for similar signed numbers
    if rad_angle < 0:
        result = -result

    return result if radians else np.rad2deg(result)


def clockwise_angle(angle: float, radians: bool = False) -> float:
    """Return the angle in the clockwise direction. Will always return an angle >= 0."""
    shift = np.pi if radians else 180
    return normalize_angle(angle, radians=radians) + shift


class Vector(np.ndarray):
    """A vector class that extends numpy.ndarray with some useful methods."""

    def __new__(cls, *args):
        """Create a new vector from a sequence of numbers"""
        if isinstance(args[0], cls):
            args = args[0].tolist()
        return np.array(args, dtype=float).view(cls)

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

    @property
    def magnitude(self) -> float:
        """Get the magnitude of a vector"""
        return np.linalg.norm(self)

    @magnitude.setter
    def magnitude(self, magnitude: float):
        """Set the magnitude of the vector"""
        if self.magnitude > 0:
            self *= magnitude / self.magnitude

    def unit(self) -> "Vector":
        """Normalize the vector"""
        copy = self.copy()
        copy.magnitude = 1.0
        return copy

    def angle(self, other=None, radians: bool = True, axis: int = 0) -> float:
        """
        Get the angle between this vector and another,
        or between this vector and the axis.
        Max angle is PI (180 degrees)
        """
        if other is None:
            other = np.zeros(self.shape).view(Vector)
            other[axis] = 1.0
        if not isinstance(other, Vector):
            other = Vector(other)
        result = np.arccos(self.dot(other) / (self.magnitude * other.magnitude))
        return result if radians else np.rad2deg(result)

    @classmethod
    def random(cls, dimensions: int) -> "Vector":
        """Create a random vector with the given number of dimensions"""
        return np.random.rand(dimensions).view(cls)


class Vector2d(Vector):
    """A Vector2d class that extends Vector with some useful methods for 2 dimensional vectors."""

    def __new__(cls, x: float = 0.0, y: float = 0.0):
        """Create a new vector2d"""
        return Vector.__new__(cls, x, y)

    def __array_ufunc__(self, ufunc, method, *args, out=None, **kwargs):
        """Try to keep the default behavior of numpy.ndarray for operations."""
        args = [x.view(np.ndarray) if isinstance(x, Vector2d) else x for x in args]
        if out:
            outs = [x.view(np.ndarray) if isinstance(x, Vector2d) else x for x in out]
            kwargs["out"] = tuple(outs)
        else:
            outs = (None,) * ufunc.nout

        results = super().__array_ufunc__(ufunc, method, *args, **kwargs)
        if results is NotImplemented:
            return NotImplemented

        if ufunc.nout == 1:
            results = (results,)

        results = tuple(
            (np.asarray(result).view(Vector2d) if out is None else out)
            for result, out in zip(results, outs)
        )

        return results[0] if len(results) == 1 else results

    @property
    def x(self) -> float:
        """Get the x component of the vector"""
        return self[0]

    @x.setter
    def x(self, value: float):
        """Set the x component of the vector"""
        self[0] = value

    @property
    def y(self) -> float:
        """Get the y component of the vector"""
        return self[1]

    @y.setter
    def y(self, value: float):
        """Set the y component of the vector"""
        self[1] = value

    @property
    def heading(self) -> float:
        """Get the heading of the vector"""
        return np.arctan2(self[1], self[0])

    def rotate(self, angle: float, radians: bool = True) -> "Vector2d":
        """Rotate the vector by the given angle, clockwise. Returns a new Vector2d."""
        if not radians:
            angle = np.deg2rad(angle)
        rotation_matrix = np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        )
        return rotation_matrix.dot(self).view(Vector2d)

    @classmethod
    def random(cls) -> "Vector2d":
        return Vector.random(2).view(cls)
