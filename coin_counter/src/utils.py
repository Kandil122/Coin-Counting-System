# ============================================================
#  utils.py
#  Shared helper functions used across all pipeline stages
# ============================================================

import cv2
import numpy as np
import os


def crop_roi(image, x, y, r, padding=5):
    """
    Crop a square region around a detected coin circle.
    Returns the cropped image patch.
    """
    h, w = image.shape[:2]
    x1 = max(0, x - r - padding)
    y1 = max(0, y - r - padding)
    x2 = min(w, x + r + padding)
    y2 = min(h, y + r + padding)
    crop = image[y1:y2, x1:x2]
    return crop


def extract_hsv_mean(crop):
    """
    Convert crop to HSV and return mean H, S, V values.
    """
    if crop is None or crop.size == 0:
        return 0.0, 0.0, 0.0
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mean_h, mean_s, mean_v, _ = cv2.mean(hsv)
    return mean_h, mean_s, mean_v


def draw_overlay(frame, circles, labels, confidences, total):
    """
    Draw detected circles, labels, confidence scores,
    and total EGP value onto the frame.
    """
    from coin_map import COIN_LABELS

    for (x, y, r), label, conf in zip(circles, labels, confidences):
        # green for known coins, red for unknown
        color = (0, 220, 80) if label != "unknown" else (0, 0, 220)
        # draw circle boundary
        cv2.circle(frame, (x, y), r, color, 2)
        # draw center dot
        cv2.circle(frame, (x, y), 3, color, -1)
        # draw label above coin
        display = COIN_LABELS.get(label, label)
        cv2.putText(
            frame,
            f"{display} {conf:.0%}",
            (x - 28, y - r - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            color,
            1,
            cv2.LINE_AA
        )

    # draw total in top-left corner
    cv2.rectangle(frame, (10, 10), (280, 58), (0, 0, 0), -1)
    cv2.putText(
        frame,
        f"Total: {total:.2f} EGP",
        (18, 44),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.1,
        (0, 220, 255),
        2,
        cv2.LINE_AA
    )
    return frame


def load_templates(templates_dir):
    """
    Load all reference template images from data/templates/.
    Returns a dict: { "EGP_1": [img, img, ...], ... }
    Each denomination folder contains up to 12 rotation images.
    """
    templates = {}
    if not os.path.exists(templates_dir):
        print(f"[WARN] Templates directory not found: {templates_dir}")
        return templates

    for label in os.listdir(templates_dir):
        label_dir = os.path.join(templates_dir, label)
        if not os.path.isdir(label_dir):
            continue
        images = []
        for fname in sorted(os.listdir(label_dir)):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                path = os.path.join(label_dir, fname)
                img = cv2.imread(path)
                if img is not None:
                    images.append(img)
        if images:
            templates[label] = images
            print(f"[INFO] Loaded {len(images)} templates for {label}")

    return templates


def ensure_dir(path):
    """Create directory if it does not exist."""
    os.makedirs(path, exist_ok=True)


def scale_circles(circles, scale):
    """
    Scale circle coordinates back after downscaling the frame.
    e.g. if detected on 0.5x frame, multiply by 2 to get original coords.
    """
    factor = int(1 / scale)
    return [(x * factor, y * factor, r * factor) for (x, y, r) in circles]