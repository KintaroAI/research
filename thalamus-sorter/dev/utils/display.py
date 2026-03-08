"""OpenCV display helpers."""

import cv2
import numpy as np


def show_grid(name, matrix):
    """Display a 2D numpy array as a normalized grayscale image."""
    normalized = cv2.normalize(
        matrix, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    cv2.imshow(name, normalized)


def show_vector(name, vector, width, height):
    """Display a flat vector as a 2D grayscale image."""
    frame = np.reshape(vector, (height, width))
    show_grid(name, frame)


def wait(ms=1):
    """Process CV2 events and wait."""
    cv2.waitKey(ms)
