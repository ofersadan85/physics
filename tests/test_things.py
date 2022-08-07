import numpy as np
from base import RotatingThing, Thing
from vectors import normalize_angle


def test_construct_thing():
    # Test default (empty) constructor
    t = Thing()
    assert t.mass == 1.0
    assert t.position.x == 0.0
    assert t.position.y == 0.0
    assert t.position.magnitude == 0.0
    assert t.position.heading == 0.0
    assert t.velocity.x == 0.0
    assert t.velocity.y == 0.0
    assert t.velocity.magnitude == 0.0
    assert t.velocity.heading == 0.0
    assert t.lock_x is False
    assert t.lock_y is False
    assert hasattr(t, "uid")

    # Test constructor with parameters
    t = Thing(position=(3.0, 4.0), velocity=(6.0, 8.0), mass=2.0)
    assert t.mass == 2
    assert t.position.x == 3
    assert t.position.y == 4
    assert t.position.magnitude == 5
    assert t.velocity.x == 6
    assert t.velocity.y == 8
    assert t.velocity.magnitude == 10
    assert hasattr(t, "uid")


def test_construct_rotating():
    # Test default (empty) constructor
    t = RotatingThing()
    assert t.mass == 1.0
    assert t.position.x == 0.0
    assert t.position.y == 0.0
    assert t.position.magnitude == 0.0
    assert t.position.heading == 0.0
    assert t.velocity.x == 0.0
    assert t.velocity.y == 0.0
    assert t.velocity.magnitude == 0.0
    assert t.velocity.heading == 0.0
    assert t.lock_x is False
    assert t.lock_y is False
    assert t.heading == 0.0
    assert t.angular_velocity == 0.0
    assert t.lock_heading is False
    assert hasattr(t, "uid")

    # Test constructor with parameters
    t = RotatingThing(
        position=(3.0, 4.0),
        velocity=(6.0, 8.0),
        mass=2.0,
        heading=np.pi,
        angular_velocity=np.pi / 2,
    )
    assert t.mass == 2
    assert t.position.x == 3
    assert t.position.y == 4
    assert t.position.magnitude == 5
    assert t.velocity.x == 6
    assert t.velocity.y == 8
    assert t.velocity.magnitude == 10
    assert t.heading == np.pi
    assert t.angular_velocity == np.pi / 2
    assert hasattr(t, "uid")


def test_update_position():
    # Test known values
    t = Thing(velocity=(6.0, 8.0))
    assert t.position.x == 0
    assert t.position.y == 0
    t.update()
    assert t.position.x == 6
    assert t.position.y == 8
    assert t.position.magnitude == 10
    t.update()
    assert t.position.magnitude == 20

    # Test negative velocity
    t = Thing(velocity=(-6.0, -8.0))
    assert t.position.x == 0
    assert t.position.y == 0
    t.update()
    assert t.position.x == -6
    assert t.position.y == -8
    assert t.position.magnitude == 10
    t.update()
    assert t.position.magnitude == 20

    # Test random values
    t = Thing.random()
    copy = t.copy()
    t.update()
    assert np.isclose(t.position.x, copy.position.x + t.velocity.x)
    assert np.isclose(t.position.y, copy.position.y + t.velocity.y)
    assert t.velocity.x == copy.velocity.x  # velocity should not change
    assert t.velocity.y == copy.velocity.y  # velocity should not change


def test_update_heading():
    # Test known values from heading = 0
    t = RotatingThing(angular_velocity=np.pi / 2)
    assert t.heading == 0
    t.update()
    assert normalize_angle(t.heading) == normalize_angle(np.pi / 2)

    # Test known values from heading = pi
    t = RotatingThing(heading=np.pi, angular_velocity=np.pi / 2)
    assert t.heading == np.pi
    t.update()
    assert normalize_angle(t.heading) == normalize_angle(np.pi + np.pi / 2)

    # Test random values
    t = RotatingThing.random()
    copy = t.copy()
    for i in range(10):
        assert np.isclose(
            t.heading, normalize_angle(copy.heading + t.angular_velocity * i)
        )
        t.update()
        # angular_velocity should not change
        assert t.angular_velocity == copy.angular_velocity

    # Check that position was updated as well
    assert np.isclose(t.position.x, copy.position.x + t.velocity.x * 10)
    assert np.isclose(t.position.y, copy.position.y + t.velocity.y * 10)
    assert t.velocity.x == copy.velocity.x  # velocity should not change
    assert t.velocity.y == copy.velocity.y  # velocity should not change


def test_copy_inequality():
    for _ in range(10):
        t = Thing.random()
        copy = t.copy()
        assert t != copy
        assert t.position.x == copy.position.x
        assert t.position.y == copy.position.y
        assert t.velocity.x == copy.velocity.x
        assert t.velocity.y == copy.velocity.y
        assert t.mass == copy.mass

        t = RotatingThing.random()
        copy = t.copy()
        assert t != copy
        assert t.position.x == copy.position.x
        assert t.position.y == copy.position.y
        assert t.velocity.x == copy.velocity.x
        assert t.velocity.y == copy.velocity.y
        assert t.mass == copy.mass
        assert t.heading == copy.heading
        assert t.angular_velocity == copy.angular_velocity
