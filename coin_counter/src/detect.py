# ============================================================
#  detect.py
#  Step 3 of pipeline — detect coins using Circular Hough Transform
# ============================================================

import cv2
import numpy as np


# ── Default HoughCircles parameters ──────────────────────────
# Tune these in notebook 03_detection.ipynb for your camera setup
HOUGH_PARAMS = {
    "dp":        1.2,   # inverse ratio of accumulator resolution
    "minDist":   40,    # min distance between detected coin centers (pixels)
    "param1":    100,   # upper Canny edge threshold
    "param2":    30,    # accumulator threshold — lower = detect more circles
    "minRadius": 20,    # smallest coin radius (25 Piastres)
    "maxRadius": 80,    # largest coin radius  (1 Pound)
}


def detect_coins(preprocessed_image, params=None):
    """
    Detect coins in a preprocessed grayscale image using
    the Circular Hough Transform (cv2.HoughCircles).

    Args:
        preprocessed_image : grayscale image from preprocess.py
        params             : dict of HoughCircles parameters (optional)

    Returns:
        list of (x, y, radius) tuples — one per detected coin
        empty list if no coins found
    """
    p = params if params else HOUGH_PARAMS

    circles = cv2.HoughCircles(
        preprocessed_image,
        cv2.HOUGH_GRADIENT,
        dp=p["dp"],
        minDist=p["minDist"],
        param1=p["param1"],
        param2=p["param2"],
        minRadius=p["minRadius"],
        maxRadius=p["maxRadius"]
    )

    if circles is None:
        return []

    circles = np.round(circles[0, :]).astype("int")
    return [(int(x), int(y), int(r)) for (x, y, r) in circles]


def check_overlap(circles):
    """
    Check whether any two detected circles overlap (coins are touching).

    Returns:
        True  — at least one pair of circles intersects
        False — all circles are separate
    """
    for i in range(len(circles)):
        for j in range(i + 1, len(circles)):
            x1, y1, r1 = circles[i]
            x2, y2, r2 = circles[j]
            dist = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            if dist < (r1 + r2):
                return True
    return False


def filter_circles(circles, image_shape):
    """
    Remove circles whose center or boundary falls outside the image.
    Avoids crop errors in later pipeline steps.
    """
    h, w = image_shape[:2]
    valid = []
    for (x, y, r) in circles:
        if (x - r >= 0 and y - r >= 0 and
                x + r < w and y + r < h):
            valid.append((x, y, r))
    return valid


def detect_and_validate(preprocessed_image, original_shape, params=None):
    """
    Full detection step:
    1. Run HoughCircles
    2. Filter out-of-bounds circles
    3. Return validated circles + overlap flag

    Args:
        preprocessed_image : grayscale image from preprocess.py
        original_shape     : shape of the original color image
        params             : optional custom HoughCircles params

    Returns:
        circles   : list of valid (x, y, r) tuples
        has_overlap: bool — True if any coins are touching
    """
    circles = detect_coins(preprocessed_image, params)
    circles = filter_circles(circles, original_shape)
    has_overlap = check_overlap(circles) if len(circles) > 1 else False
    return circles, has_overlap