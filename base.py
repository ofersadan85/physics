import uuid
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from vectors import Vector2d, clockwise_angle, normalize_angle


@dataclass
class Thing:  # Currently only used in 2D
    """A thing is any object with changing position and mass"""

    position: Sequence = field(default_factory=Vector2d)
    velocity: Sequence = field(default_factory=Vector2d)
    mass: float = 1.0
    lock_x: bool = False
    lock_y: bool = False
    uid: uuid.UUID = field(default_factory=uuid.uuid4, init=False)

    def __post_init__(self):
        if not isinstance(self.position, Vector2d):
            self.position = Vector2d(*self.position)
        if not isinstance(self.velocity, Vector2d):
            self.velocity = Vector2d(*self.velocity)
        self.mass = abs(self.mass)

    def __hash__(self) -> int:
        return hash(self.uid)

    def __eq__(self, other) -> bool:
        return isinstance(other, Thing) and self.uid == other.uid

    def apply_force(self, force: Vector2d) -> None:
        """Apply a force to the thing"""
        self.velocity = self.velocity + force / self.mass

    def distance(self, other: "Thing") -> float:
        """Calculate the distance between two things"""
        return np.linalg.norm(self.position - other.position)

    def update(self):
        """Update position"""
        lock = [not self.lock_x, not self.lock_y]
        self.velocity = self.velocity * lock
        self.position = self.position + self.velocity

    def bounce(self, axis: int = 0, ratio: float = 1.0) -> None:
        """
        Bounce off something in a given axis, with a given ratio velocity
        i.e. if ratio is 0.99, the velocity will be reversed and reduced by 1%
        """
        self.velocity[axis] = -self.velocity[axis] * ratio

    def copy(self) -> "Thing":
        """Create a copy of this thing"""
        return Thing(
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            mass=self.mass,
            lock_x=self.lock_x,
            lock_y=self.lock_y,
        )

    @classmethod
    def random(cls) -> "Thing":
        """Create a random thing"""
        return Thing(
            position=Vector2d.random(),
            velocity=Vector2d.random(),
            mass=np.random.rand(),
        )


@dataclass
class RotatingThing(Thing):
    """A Thing object that has a heading and angular_velocity"""

    heading: float = 0.0
    angular_velocity: float = 0.0
    lock_heading: bool = False

    def __post_init__(self):
        super().__post_init__()
        self.heading = normalize_angle(self.heading)

    def __hash__(self) -> int:
        return hash(self.uid)

    def __eq__(self, other) -> bool:
        return isinstance(other, Thing) and self.uid == other.uid

    def apply_angular_force(self, force: float) -> None:
        """Apply an angular force to the thing"""
        self.angular_velocity += force / self.mass

    def update(self):
        """Update position and heading"""
        super().update()
        if self.lock_heading:
            self.angular_velocity = 0.0
        self.heading += self.angular_velocity
        self.heading = normalize_angle(self.heading)

    def copy(self) -> "RotatingThing":
        """Create a copy of this thing"""
        return RotatingThing(
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            mass=self.mass,
            heading=self.heading,
            angular_velocity=self.angular_velocity,
            lock_x=self.lock_x,
            lock_y=self.lock_y,
            lock_heading=self.lock_heading,
        )

    @classmethod
    def random(cls) -> "RotatingThing":
        """Create a random RotatingThing"""
        return RotatingThing(
            position=Vector2d.random(),
            velocity=Vector2d.random(),
            mass=np.random.rand(),
            heading=(np.random.rand() * 2 * np.pi),
            angular_velocity=np.random.rand(),
        )


@dataclass
class Spring:
    """A spring between two things"""

    a: Thing = field(default_factory=Thing)
    b: Thing = field(default_factory=Thing)
    k: float = 0.1
    rest_length: float = 1.0

    def update(self):
        """Apply a spring force between the two things"""
        force = self.a.position - self.b.position  # Vector from b to a
        force.magnitude = (self.rest_length - force.magnitude) * self.k
        self.a.apply_force(-force)
        self.b.apply_force(force)
