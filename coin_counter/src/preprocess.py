# ============================================================
#  preprocess.py
#  Step 2 of pipeline — prepare image before Hough detection
# ============================================================

import cv2
import numpy as np


def to_grayscale(image):
    """Convert BGR image to grayscale."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def apply_blur(gray, kernel_size=(9, 9), sigma=2):
    """
    Apply Gaussian blur to suppress noise.
    Larger kernel = smoother but less detail.
    Recommended: (9,9) for static images, (5,5) for real-time.
    """
    return cv2.GaussianBlur(gray, kernel_size, sigma)


def apply_clahe(gray, clip_limit=2.0, tile_size=(8, 8)):
    """
    Contrast Limited Adaptive Histogram Equalization.
    Improves contrast on reflective Egyptian coin surfaces.
    Especially useful for the gold 1-Pound coin under uneven lighting.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    return clahe.apply(gray)


def preprocess(image, use_clahe=True, blur_kernel=(9, 9)):
    """
    Full preprocessing pipeline for a single frame or image.
    Steps: BGR → grayscale → CLAHE → Gaussian blur

    Args:
        image      : BGR image (numpy array)
        use_clahe  : apply adaptive contrast (recommended True)
        blur_kernel: Gaussian blur kernel size

    Returns:
        preprocessed grayscale image ready for HoughCircles
    """
    gray = to_grayscale(image)

    if use_clahe:
        gray = apply_clahe(gray)

    blurred = apply_blur(gray, kernel_size=blur_kernel)
    return blurred


def resize_for_detection(image, scale=0.5):
    """
    Downscale image before running HoughCircles for real-time speed.
    Detect at 0.5x resolution → 4x faster → scale coords back with scale_circles().
    """
    h, w = image.shape[:2]
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def preprocess_crop(crop):
    """
    Lightweight preprocessing for an individual coin crop.
    Used before morphological operations in Steps 4-5.
    """
    if crop is None or crop.size == 0:
        return None
    # resize to standard size for consistent feature extraction
    resized = cv2.resize(crop, (64, 64), interpolation=cv2.INTER_AREA)
    return resized