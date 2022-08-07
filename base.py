import uuid
from dataclasses import dataclass, field

import numpy as np

from vectors import Vector, Vector2d, normalize_angle

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
        self.heading = normalize_angle(self.heading)
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
        self.heading = normalize_angle(self.heading)

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
