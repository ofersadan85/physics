import time

import cv2 as cv
import numpy as np

from base import Spring, Thing
from vectors import Vector2d

start = time.time()
frame_count = 0
fps = 0
font = cv.FONT_HERSHEY_SIMPLEX
CANVAS_SIZE = (512, 512)
WINDOW_NAME = "My Canvas"
RADIUS = 10
THINGS = []  # type: list[Thing]
springs = []  # type: list[Spring]


def mouse_handler(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        t = Thing.random()
        t.position = Vector2d(x, y)
        if len(THINGS) == 0:
            t.lock_x = True
            t.lock_y = True
        if len(THINGS) > 0:
            t.mass = THINGS[-1].mass * 2
            s = Spring(
                t, np.random.choice(THINGS), k=0.001, rest_length=1.0, dampening=0.001
            )
            springs.append(s)
        THINGS.append(t)


if __name__ == "__main__":
    cv.namedWindow(WINDOW_NAME, cv.WINDOW_AUTOSIZE)
    cv.setMouseCallback(WINDOW_NAME, mouse_handler)
    shape = CANVAS_SIZE + (3,)
    gravity = Vector2d(0, 1)
    gravity.magnitude = 0.01
    while True:
        frame_count += 1
        canvas = np.zeros(shape, dtype=np.uint8)

        for spring in springs:
            spring.update()
            cv.line(
                canvas,
                spring.a.position.astype(int),
                spring.b.position.astype(int),
                (0, 255, 0),
                1,
                cv.LINE_AA,
            )

        for t in THINGS:
            # Bounce off the walls
            x, y = t.position.astype(int)
            if x <= 0 or x >= CANVAS_SIZE[1]:
                t.bounce(axis=0, ratio=0.5)
                t.position.x = np.clip(t.position.x, 1, CANVAS_SIZE[1] - 1)
            if y <= 0 or y >= CANVAS_SIZE[0]:
                t.bounce(axis=1, ratio=0.5)
                t.position.y = np.clip(t.position.y, 1, CANVAS_SIZE[0] - 1)

            # Add gravity
            t.apply_force(gravity * t.mass)

            # Update and draw
            t.update()
            cv.circle(
                canvas,
                t.position.astype(int),
                int(np.sqrt(t.mass)),
                (255, 255, 255),
                -1,
                cv.LINE_AA,
            )
            next_position = t.position + t.velocity * 10
            gravity_acceleration = t.position + gravity * 1500
            cv.arrowedLine(
                canvas,
                t.position.astype(int),
                next_position.astype(int),
                (0, 255, 255),
                1,
                cv.LINE_AA,
            )
            cv.arrowedLine(
                canvas,
                t.position.astype(int),
                gravity_acceleration.astype(int),
                (0, 0, 255),
                1,
                cv.LINE_AA,
            )

        if frame_count % 100 == 0:
            fps = 1 / (time.time() - start)
            start = time.time()

        fps_text = f"fps: {(fps * 100):.2f}"
        cv.putText(canvas, fps_text, (10, 30), font, 0.5, (100, 255, 0), 1, cv.LINE_AA)

        cv.imshow(WINDOW_NAME, canvas)

        if cv.waitKey(1) == 27:
            cv.destroyWindow(WINDOW_NAME)
            break
