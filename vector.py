import math

import numpy as np

# Shorthand variables
PI = math.pi
TAU = math.tau
TWO_PI = TAU
RIGHT_ANGLE = PI / 2


def set_angle_mode(mode: int):
    """Set the angle mode"""
    global _ANGLE_MODE
    _ANGLE_MODE = int(bool(mode))


def magnitude(v: np.ndarray) -> float:
    """Get the magnitude of a vector"""
    return np.linalg.norm(v)


def normalize(v: np.ndarray) -> np.ndarray:
    """Normalize the vector v - Get the unit vector of v"""
    return v / magnitude(v)


def radians(degrees: float) -> float:
    """Convert degrees to radians"""
    return degrees * PI / 180.0


def degrees(radians: float) -> float:
    """Convert radians to degrees"""
    return radians * 180.0 / PI


def angle(v1: np.ndarray, v2: np.ndarray | None = None, radians: bool = True) -> float:
    """Get the angle between two vectors, or between a vector and the x-axis. Max angle is PI (180 degrees)"""
    if v2 is None:
        v2 = np.zeros(np.array(v1).shape)
        v2[0] = 1.0
    result = np.arccos(np.dot(v1, v2) / (magnitude(v1) * magnitude(v2)))
    return result if radians else degrees(result)
