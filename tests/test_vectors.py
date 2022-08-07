from itertools import product

import numpy as np
from vectors import Vector, Vector2d, clockwise_angle, normalize_angle


def test_normalize_angle():
    for angle in range(-1000, 1000):
        deg = normalize_angle(angle, radians=False)
        rad = normalize_angle(np.deg2rad(angle), radians=True)
        assert -np.pi < rad <= np.pi
        assert -180 < deg <= 180
        # Checking with np.isclose is used because of floating point inaccuracies
        assert np.isclose(deg, np.rad2deg(rad))
        if -180 < angle <= 180:
            assert np.isclose(angle, np.rad2deg(rad))
            # Checking exact equality and not np.isclose to check it was not modified
            assert angle == deg


def test_v2d_get_set():
    v = Vector2d(1, 2)
    assert v.x == 1
    assert v.y == 2
    v.x = 3
    v.y = 4
    assert v.x == 3
    assert v.y == 4
    assert v.magnitude == 5
    v.magnitude = 10
    assert v.x == 6
    assert v.y == 8
    u = v.unit()
    assert u.magnitude == 1


def test_heading_magnitude():
    vectors = [Vector2d(x, y) for x, y in product((-1, 0, 1), (-1, 0, 1))]
    headings = [
        -0.75 * np.pi,
        np.pi,
        0.75 * np.pi,
        -0.5 * np.pi,
        0.0,
        0.5 * np.pi,
        -0.25 * np.pi,
        0.0,
        0.25 * np.pi,
    ]

    for v, h in zip(vectors, headings):
        assert v.heading == h

        # Magnitude tests
        if v.x == 0 and v.y == 0:
            # Zero vector has no magnitude
            assert v.magnitude == 0
        elif 0 in (v.x, v.y):
            # Only one component is non-zero
            assert v.magnitude == 1
        else:
            # Both components are non-zero
            assert v.magnitude == np.sqrt(2)
            assert np.isclose(v.unit().magnitude, 1)

    for x, y in product([3, -3], [4, -4]):
        assert Vector2d(x, y).magnitude == 5.0


def test_rotations():
    zero = Vector2d(0, 0)
    for angle in range(-360, 361):
        # Test rotations for the zero vector
        assert zero.x == 0.0
        assert zero.y == 0.0
        assert zero.rotate(angle, radians=False).magnitude == 0.0
        assert zero.rotate(angle, radians=False).heading == 0.0

        # Test different ways to rotate by 180 for a random vector
        v_random = Vector2d.random()
        opposite1 = v_random.rotate(np.pi)
        opposite2 = v_random.rotate(180, radians=False)
        opposite3 = -v_random
        for neg in (opposite1, opposite2, opposite3):
            assert np.isclose(v_random.magnitude, neg.magnitude)
            assert np.isclose(abs(v_random.heading) + abs(neg.heading), np.pi)

        # Test rotations by current tested angle for a random vector
        v_random = Vector2d.random()
        angle = np.deg2rad(clockwise_angle(angle))  # Clockwise rotation easier to test
        rotated = v_random.rotate(angle)
        assert np.isclose(v_random.magnitude, rotated.magnitude)
        # TODO: Check that the angle is correct
