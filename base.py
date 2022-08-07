import uuid
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from vectors import Vector2d, normalize_angle


@dataclass
class Thing:  # Currently only used in 2D
    """A thing is any object with changing position and mass"""

    mass: float = 1.0
    position: Sequence = field(default_factory=Vector2d)
    velocity: Sequence = field(default_factory=Vector2d)
    lock_x: bool = False
    lock_y: bool = False
    uid: uuid.UUID = field(default_factory=uuid.uuid4)

    def __post_init__(self):
        self.mass = abs(self.mass)
        if not isinstance(self.position, Vector2d):
            self.position = Vector2d(*self.position)
        if not isinstance(self.velocity, Vector2d):
            self.velocity = Vector2d(*self.velocity)

    def __hash__(self) -> int:
        return hash(self.uid)

    def __eq__(self, other) -> bool:
        return isinstance(other, Thing) and self.uid == other.uid

    def distance(self, other: "Thing") -> float:
        """Calculate the distance between two things"""
        return np.linalg.norm(self.position - other.position)

    def update(self):
        """Update position"""
        self.velocity *= [self.lock_x, self.lock_y]
        self.position += self.velocity

    def bounce(self, axis: int = 0, ratio: float = 1.0) -> None:
        """
        Bounce off something in a given axis, with a given ratio velocity
        i.e. if ratio is 0.99, the velocity will be reversed and reduced by 1%
        """
        self.velocity[axis] = -self.velocity[axis] * ratio


@dataclass
class RotatingThing(Thing):
    """A Thing object that has a heading and angular_velocity"""

    heading: float = 0.0
    angular_velocity: float = 0.0
    lock_heading: bool = False

    def __post_init__(self):
        super().__post_init__()
        self.heading = normalize_angle(self.heading)

    def update(self):
        """Update position and heading"""
        super().update()
        if self.lock_heading:
            self.angular_velocity = 0.0
        self.heading += self.angular_velocity
        self.heading = normalize_angle(self.heading)
