"""Camera capture utilities."""

import cv2
import numpy as np


class Camera:
    """Wraps cv2.VideoCapture for grayscale frame capture."""

    def __init__(self, width, height, device=0):
        self.width = width
        self.height = height
        self.dim = (width, height)
        self.cap = cv2.VideoCapture(device)

    def read_gray(self):
        """Capture one frame, resize, convert to grayscale, return as
        flat float vector normalized to [0, 1]."""
        while True:
            ret, frame = self.cap.read()
            if ret:
                break
        resized = cv2.resize(frame, self.dim)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        return np.ravel(gray) / 255.0

    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.release()
