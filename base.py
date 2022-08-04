import uuid
from dataclasses import dataclass, field

import numpy as np

from vector import Vector2d

""" This file defines base classes and constants"""


@dataclass
class Thing:
    """A thing is any object with position and mass"""

    mass: float = 1.0
    position: Vector2d = field(default_factory=Vector2d)
    velocity: Vector2d = field(default_factory=Vector2d)
    acceleration: Vector2d = field(default_factory=Vector2d)
    heading: float = 0.0
    angular_velocity: float = 0.0
    angular_acceleration: float = 0.0
    uuid: uuid.UUID = field(default_factory=uuid.uuid4)

    def __post_init__(self):
        """Make sure the position is a vector and that mass is positive"""
        self.mass = abs(self.mass)
        self.position = Vector2d(self.position)

    def __hash__(self) -> int:
        return hash(self.uuid)

    def __eq__(self, other) -> bool:
        return isinstance(other, Thing) and self.uuid == other.uuid

    def distance(self, other: "Thing") -> float:
        """Calculate the distance between two things"""
        return np.linalg.norm(self.position - other.position)

    def update(self):
        """Update velocity, position and heading"""
        self.velocity += self.acceleration
        self.position += self.velocity
        self.angular_velocity += self.angular_acceleration
        self.heading += self.angular_velocity

    def bounce(self, x: bool = True, ratio: float = 1.0) -> None:
        """
        Bounce off something in a given axis, with a given ratio velocity
        i.e. if ratio is 0.99, the velocity will be reversed and reduced by 1%
        """
        axis = 0 if x else 1
        self.position[axis] = -self.position[axis] * ratio

    def flip(self) -> None:
        """Flip the velocity and heading in all directions"""
        self.velocity = -self.velocity
        self.heading = -self.heading
