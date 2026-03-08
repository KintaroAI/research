"""OpenCV display helpers. Gracefully degrade when no GUI is available."""

import os
import cv2
import numpy as np

# Detect headless environment
_HEADLESS = not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY")


def show_grid(name, matrix):
    """Display a 2D numpy array as a normalized grayscale image."""
    if _HEADLESS:
        return
    normalized = cv2.normalize(
        matrix, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    cv2.imshow(name, normalized)


def show_vector(name, vector, width, height):
    """Display a flat vector as a 2D grayscale image."""
    frame = np.reshape(vector, (height, width))
    show_grid(name, frame)


def wait(ms=1):
    """Process CV2 events and wait."""
    if _HEADLESS:
        return
    cv2.waitKey(ms)


def poll_quit():
    """Check if user pressed 'q'. Returns True to quit."""
    if _HEADLESS:
        return False
    return cv2.waitKey(1) & 0xFF == ord('q')
