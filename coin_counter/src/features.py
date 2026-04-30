# ============================================================
#  features.py
#  Step 6 of pipeline — extract features from each coin crop
#  Features: radius, mean HSV, HSV histogram, texture
# ============================================================

import cv2
import numpy as np


def extract_color_features(crop):
    """
    Extract color-based features from a coin crop.

    Returns:
        mean_h  : mean Hue value
        mean_s  : mean Saturation value
        mean_v  : mean Value (brightness)
        hist    : normalized HSV histogram (512-dim flattened vector)
    """
    if crop is None or crop.size == 0:
        return 0.0, 0.0, 0.0, np.zeros(512)

    # resize to standard size for consistent histograms
    resized = cv2.resize(crop, (64, 64), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)

    # mean channel values
    mean_h, mean_s, mean_v, _ = cv2.mean(hsv)

    # 3D HSV histogram (8 bins per channel = 8x8x8 = 512 values)
    hist = cv2.calcHist(
        [hsv],
        [0, 1, 2],          # H, S, V channels
        None,               # no mask
        [8, 8, 8],          # bins per channel
        [0, 180, 0, 256, 0, 256]  # ranges
    )
    hist = cv2.normalize(hist, hist).flatten()

    return mean_h, mean_s, mean_v, hist


def extract_texture_features(crop):
    """
    Extract texture feature using Laplacian variance.
    Higher variance = more texture/detail (e.g. coin engravings).
    Lower variance = smoother surface.

    This helps distinguish coins with different surface finishes.

    Returns:
        laplacian_var : float — texture intensity measure
    """
    if crop is None or crop.size == 0:
        return 0.0

    resized = cv2.resize(crop, (64, 64), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var


def extract_shape_features(crop, radius):
    """
    Extract shape-based features from a coin crop.

    Returns:
        circularity : how close the detected region is to a perfect circle
        area_ratio  : ratio of coin area to bounding box area
    """
    if crop is None or crop.size == 0:
        return 0.0, 0.0

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0, 0.0

    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)

    # circularity: 1.0 = perfect circle
    circularity = (4 * np.pi * area / (perimeter ** 2)) if perimeter > 0 else 0.0

    # area ratio: coin area vs bounding square area
    bbox_area = (2 * radius) ** 2
    area_ratio = area / bbox_area if bbox_area > 0 else 0.0

    return circularity, area_ratio


def extract_features(crop, radius):
    """
    Full feature extraction for one coin crop.
    Combines color, texture, and shape into a single feature vector.

    Used by:
    - classify.py  for individual feature votes
    - svm_classifier.py as input to the SVM

    Args:
        crop   : BGR coin crop (from utils.crop_roi)
        radius : detected circle radius in pixels

    Returns:
        numpy array of shape (517,) containing:
            [0]       radius
            [1-3]     mean H, S, V
            [4-515]   HSV histogram (512 values)
            [516]     Laplacian texture variance
    """
    mean_h, mean_s, mean_v, hist = extract_color_features(crop)
    texture = extract_texture_features(crop)

    feature_vector = np.concatenate([
        [float(radius)],               # 1  — size
        [mean_h, mean_s, mean_v],      # 3  — mean color
        hist.astype(np.float32),       # 512 — color distribution
        [texture],                     # 1  — texture
    ])

    return feature_vector.astype(np.float32)


def get_feature_names():
    """
    Returns human-readable names for each feature dimension.
    Useful for analysis in notebooks.
    """
    names = ["radius", "mean_H", "mean_S", "mean_V"]
    for i in range(512):
        names.append(f"hist_{i}")
    names.append("texture_var")
    return names