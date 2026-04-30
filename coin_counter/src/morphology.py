# ============================================================
#  morphology.py
#  Step 5 of pipeline — clean coin masks, handle overlapping coins
# ============================================================

import cv2
import numpy as np


def clean_coin_mask(crop):
    """
    Apply morphological operations to clean a coin crop binary mask.

    Operations:
    - Otsu thresholding  : binarize the coin crop
    - Closing (CLOSE)    : fill small holes inside the coin region
    - Opening (OPEN)     : remove small noise outside the coin

    Args:
        crop : BGR coin crop from utils.crop_roi()

    Returns:
        cleaned binary mask (same spatial size as crop)
    """
    if crop is None or crop.size == 0:
        return None

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    # Otsu's thresholding — automatically finds optimal threshold
    _, binary = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # elliptical kernel matches coin shape better than square
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    # closing: close small holes inside coin region
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

    # opening: remove small noise blobs outside coin
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=1)

    return opened


def apply_mask_to_crop(crop, mask):
    """
    Apply a binary mask to a coin crop.
    Sets background pixels to black, keeping only the coin region.
    """
    if mask is None:
        return crop
    masked = cv2.bitwise_and(crop, crop, mask=mask)
    return masked


def watershed_separation(image, circles):
    """
    Apply watershed algorithm to separate overlapping coins.
    Called when check_overlap() returns True in detect.py.

    Args:
        image   : original BGR image (full frame)
        circles : list of (x, y, r) from HoughCircles

    Returns:
        markers : labeled image where each coin region has a unique integer ID
                  background = 1, boundaries = -1, coins = 2, 3, 4, ...
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # threshold to get binary foreground (coins)
    _, binary = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    kernel = np.ones((3, 3), np.uint8)

    # sure background: dilate to expand coin regions slightly
    sure_bg = cv2.dilate(binary, kernel, iterations=3)

    # sure foreground: distance transform + threshold
    dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(
        dist_transform,
        0.5 * dist_transform.max(),
        255,
        0
    )
    sure_fg = np.uint8(sure_fg)

    # unknown region = sure_bg minus sure_fg
    unknown = cv2.subtract(sure_bg, sure_fg)

    # label connected components in sure foreground
    _, markers = cv2.connectedComponents(sure_fg)

    # background gets label 1 (not 0, which watershed treats as unknown)
    markers = markers + 1

    # mark unknown region as 0
    markers[unknown == 255] = 0

    # run watershed — boundaries become -1
    markers = cv2.watershed(image, markers)

    return markers


def extract_coin_regions_from_markers(image, markers):
    """
    After watershed, extract bounding boxes for each labeled region
    and return them as individual coin crops.

    Args:
        image   : original BGR image
        markers : output of watershed_separation()

    Returns:
        list of (crop, x, y, r) tuples — approximate circle params from bbox
    """
    coins = []
    unique_labels = np.unique(markers)

    for label in unique_labels:
        # skip background (1) and boundaries (-1)
        if label <= 1:
            continue

        # create mask for this coin region
        mask = np.zeros(markers.shape, dtype=np.uint8)
        mask[markers == label] = 255

        # find bounding box
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            continue

        cnt = max(contours, key=cv2.contourArea)
        (cx, cy), radius = cv2.minEnclosingCircle(cnt)
        cx, cy, radius = int(cx), int(cy), int(radius)

        # crop using approximate circle
        x1 = max(0, cx - radius)
        y1 = max(0, cy - radius)
        x2 = min(image.shape[1], cx + radius)
        y2 = min(image.shape[0], cy + radius)
        crop = image[y1:y2, x1:x2]

        if crop.size > 0:
            coins.append((crop, cx, cy, radius))

    return coins


def erode(image, kernel_size=3, iterations=1):
    """Standalone erosion — shrinks bright regions."""
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.erode(image, kernel, iterations=iterations)


def dilate(image, kernel_size=3, iterations=1):
    """Standalone dilation — expands bright regions."""
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.dilate(image, kernel, iterations=iterations)