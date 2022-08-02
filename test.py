import numpy as np
import cv2 as cv

CANVAS_SIZE = (512, 512)
WINDOW_NAME = "My Canvas"
canvas = np.zeros(CANVAS_SIZE, dtype=np.uint8)
last_shape = np.array([0, 0])

def mouse_handler(event, x, y, flags, param):
    global last_shape
    mouse_vector = np.array([x, y])
    # print(event, x, y, flags, param)
    if event == cv.EVENT_LBUTTONDOWN:
        cv.circle(canvas, (x, y), 10, (255, 0, 0), -1, cv.LINE_AA)
        last_shape = mouse_vector
    elif event == cv.EVENT_RBUTTONDOWN:
        cv.arrowedLine(canvas, last_shape, mouse_vector, (255, 0, 0), 1, cv.LINE_AA, 0, 0.1)
        last_shape = mouse_vector
    elif event == cv.EVENT_MBUTTONDOWN:
        cv.drawMarker(canvas, (x, y), (255, 0, 0), cv.MARKER_TILTED_CROSS, 10, 1, cv.LINE_AA)
        last_shape = mouse_vector


cv.namedWindow(WINDOW_NAME, cv.WINDOW_AUTOSIZE)
cv.setMouseCallback(WINDOW_NAME, mouse_handler)
# cv.setWindowTitle(WINDOW_NAME, "My Canvas")
# cv.getWindowProperty()


def show_random():
    while True:
        img = np.random.randint(
            0, 255, size=(CANVAS_SIZE[1], CANVAS_SIZE[0], 3), dtype=np.uint8
        )
        cv.imshow(WINDOW_NAME, img)
        if cv.waitKey(1) in (27, ord("q")):
            print(cv.getWindowImageRect(WINDOW_NAME))
            break  # ESC or 'q' to quit


def show_webcam(mirror=False):
    cam = cv.VideoCapture(0)
    while True:
        ret_val, img = cam.read()
        if mirror:
            img = cv.flip(img, 1)
        cv.imshow("my webcam", img)
        if cv.waitKey(1) == 27:
            break  # esc to quit
    cv.destroyAllWindows()


def main():
    while cv.getWindowProperty(WINDOW_NAME, 0) >= 0:
        cv.imshow(WINDOW_NAME, canvas)
        if cv.waitKey(1) == 27:
            break


if __name__ == "__main__":
    main()
