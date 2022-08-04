import uuid
from dataclasses import dataclass, field

import numpy as np


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

    def magnitude(self) -> float:
        """Get the magnitude of a vector"""
        return np.linalg.norm(self)

    def normalize(self) -> "Vector":
        """Normalize the vector v - Get the unit vector of v"""
        return self / self.magnitude(self)

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
        result = np.arccos(self.dot(other) / (self.magnitude() * other.magnitude()))
        # result = np.arccos(np.dot(self, other) / (self.magnitude() * v2.magnitude()))
        return result if radians else np.rad2deg(result)

    def with_magnitude(self, magnitude: float) -> "Vector":
        """Set the magnitude of the vector"""
        return self * (magnitude / self.magnitude())

    @classmethod
    def random(cls, dimensions: int) -> "Vector":
        """Create a random vector with the given number of dimensions"""
        return cls(np.random.rand(dimensions))


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

    def heading(self) -> float:
        """Get the heading of the vector"""
        return np.arctan2(self[1], self[0])

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

    def rotate(self, angle: float, radians: bool = True) -> "Vector":
        """Set the angle of the vector"""
        if not radians:
            angle = np.deg2rad(angle)
        rotation_matrix = np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        )
        return rotation_matrix.dot(self).view(Vector2d)

    @classmethod
    def random(cls) -> "Vector2d":
        return Vector.random(2).view(Vector2d)


@dataclass
class Thing:
    """A thing is any object with position and mass"""

    mass: float = 1.0
    position: Vector = field(default_factory=Vector2d)
    velocity: Vector = field(default_factory=Vector2d)
    heading: float = 0.0
    angular_velocity: float = 0.0
    locked: list[bool] = field(default_factory=list)
    uid: uuid.UUID = field(default_factory=uuid.uuid4)

    def __post_init__(self):
        self.mass = abs(self.mass)
        self.wrap_heading()
        # Do not lock an axis if it is not specified
        # self.locked needs to be a list with the same length as the number of position dimensions + 1
        # so that self.locked[-1] represents the locked state of rotation "axis"
        while len(self.locked) < len(self.position) + 1:
            self.locked.append(False)

    def __hash__(self) -> int:
        return hash(self.uid)

    def __eq__(self, other) -> bool:
        return isinstance(other, Thing) and self.uid == other.uid

    def distance(self, other: "Thing") -> float:
        """Calculate the distance between two things"""
        return np.linalg.norm(self.position - other.position)

    def update(self):
        """Update position and heading"""
        for i, lock in enumerate(self.locked[:-1]):
            if lock:
                self.velocity[i] = 0.0
        self.position += self.velocity

        if self.locked[-1]:
            self.angular_velocity = 0.0
        self.heading += self.angular_velocity
        self.wrap_heading()

    def wrap_heading(self):
        """Wrap the heading to be -PI < heading <= PI"""
        self.heading = (self.heading + np.pi) % (2 * np.pi) - np.pi

    def lock(self, axes: int | list[int] | None = None, unlock: bool = False):
        """Lock the given axes by index, or all axes if axes is None. index of -1 locks rotation"""
        if axes is None:
            axes = range(-1, len(self.position))
        if isinstance(axes, int):
            axes = [axes]
        for index in axes:
            self.locked[index] = not unlock

    def bounce(self, axis: int = 0, ratio: float = 1.0) -> None:
        """
        Bounce off something in a given axis, with a given ratio velocity
        i.e. if ratio is 0.99, the velocity will be reversed and reduced by 1%
        """
        self.position[axis] = -self.position[axis] * ratio

    def flip(self) -> None:
        """Flip the velocity and heading in all directions"""
        self.velocity = -self.velocity
        self.heading = -self.heading
