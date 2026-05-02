# # ============================================================
# #  main.py — Coin Counter (EGP)
# #  CSE483: Computer Vision — Ain Shams University
# #
# #  Denominations: 1 EGP (bimetallic) | 50 PT (fully brass)
# #
# #  Dataset: /home/adham/Documents/Coin-Counting-System/EGP Coin Dataset/
# #
# #  USAGE:
# #    python main.py --mode calibrate
# #    python main.py --mode image    --input path/to/photo.jpg
# #    python main.py --mode realtime
# #    python main.py --mode train
# #    python main.py --mode realtime --svm   (after training)
# #
# #  INSTALL:
# #    pip install opencv-python numpy scikit-learn joblib matplotlib
# # ============================================================

# import cv2
# import numpy as np
# import os
# import sys
# import time
# import argparse
# import json
# from collections import deque


# # ════════════════════════════════════════════════════════════
# #  DATASET & PROJECT PATHS
# # ════════════════════════════════════════════════════════════

# DATASET_PATH  = "/home/adham/Documents/Coin-Counting-System/EGP Coin Dataset/batch1"
# TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), "data", "templates")
# MODELS_DIR    = os.path.join(os.path.dirname(__file__), "models")
# OUTPUTS_DIR   = os.path.join(os.path.dirname(__file__), "outputs")
# MODEL_PATH    = os.path.join(MODELS_DIR, "svm_coin_classifier.pkl")


# # ════════════════════════════════════════════════════════════
# #  STEP 1 — COIN MAP
# #  Only 2 denominations: 1 EGP and 50 PT
# #
# #  Key visual difference:
# #    1 EGP  → bimetallic (gold center + silver ring) → LOW saturation mean
# #    50 PT  → fully brass (no ring)                 → HIGH saturation mean
# #
# #  Update COIN_SIZES after running: python main.py --mode calibrate
# # ════════════════════════════════════════════════════════════

# COIN_SIZES = {
#     "EGP_1":    52,    # 1 Pound     — 23mm real diameter
#     "EGP_0.50": 45,    # 50 Piastres — 21mm real diameter
# }

# # Saturation is the key discriminator:
# #   1 EGP  → s~120  (silver ring pulls mean saturation down)
# #   50 PT  → s~160  (fully brass = consistently high saturation)
# COIN_COLORS = {
#     "EGP_1":    {"h": 20, "s": 120, "v": 170},
#     "EGP_0.50": {"h": 22, "s": 160, "v": 155},
# }

# COIN_VALUES = {
#     "EGP_1":    1.00,
#     "EGP_0.50": 0.50,
# }

# COIN_LABELS = {
#     "EGP_1":    "1 EGP",
#     "EGP_0.50": "50 PT",
#     "unknown":  "???",
# }

# ALL_COINS = list(COIN_VALUES.keys())

# # Map dataset folder names → denomination label
# # Edit this if your folders are named differently
# DATASET_LABEL_MAP = {
#     # 1 EGP variants
#     "1pound":    "EGP_1",
#     "1_pound":   "EGP_1",
#     "egp1":      "EGP_1",
#     "1egp":      "EGP_1",
#     "pound":     "EGP_1",
#     "batch1":    "EGP_1",   # your dataset: batch1 = 1 EGP
#     # 50 PT variants
#     "50pt":      "EGP_0.50",
#     "50_pt":     "EGP_0.50",
#     "50piastre": "EGP_0.50",
#     "halfpound": "EGP_0.50",
#     "batch2":    "EGP_0.50",  # your dataset: batch2 = 50 PT
#     "egp050":    "EGP_0.50",
# }


# # ════════════════════════════════════════════════════════════
# #  STEP 2 — PREPROCESSING
# # ════════════════════════════════════════════════════════════

# def preprocess(image, use_clahe=True, blur_kernel=(9, 9)):
#     """
#     BGR → preprocessed grayscale ready for HoughCircles.
#     grayscale → CLAHE → Gaussian blur

#     CLAHE is especially important here because the silver ring
#     of 1 EGP reflects light differently than the brass 50 PT.
#     """
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     if use_clahe:
#         clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
#         gray  = clahe.apply(gray)
#     blurred = cv2.GaussianBlur(gray, blur_kernel, 2)
#     return blurred


# def resize_for_detection(image, scale=0.5):
#     """Downscale for faster real-time HoughCircles."""
#     h, w = image.shape[:2]
#     return cv2.resize(image, (int(w * scale), int(h * scale)),
#                       interpolation=cv2.INTER_AREA)


# # ════════════════════════════════════════════════════════════
# #  STEP 3 — DETECTION (HOUGH TRANSFORM)
# # ════════════════════════════════════════════════════════════

# HOUGH_PARAMS = {
#     "dp":        1.2,
#     "minDist":   40,
#     "param1":    100,
#     "param2":    30,
#     "minRadius": 20,
#     "maxRadius": 80,
# }


# def detect_coins(preprocessed_image, params=None):
#     """
#     Detect coins using Circular Hough Transform.
#     Returns list of (x, y, radius) tuples.
#     """
#     p = params if params else HOUGH_PARAMS
#     circles = cv2.HoughCircles(
#         preprocessed_image,
#         cv2.HOUGH_GRADIENT,
#         dp=p["dp"],
#         minDist=p["minDist"],
#         param1=p["param1"],
#         param2=p["param2"],
#         minRadius=p["minRadius"],
#         maxRadius=p["maxRadius"],
#     )
#     if circles is None:
#         return []
#     circles = np.round(circles[0, :]).astype("int")
#     return [(int(x), int(y), int(r)) for (x, y, r) in circles]


# def check_overlap(circles):
#     """Return True if any two detected circles intersect."""
#     for i in range(len(circles)):
#         for j in range(i + 1, len(circles)):
#             x1, y1, r1 = circles[i]
#             x2, y2, r2 = circles[j]
#             dist = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
#             if dist < (r1 + r2):
#                 return True
#     return False


# def filter_circles(circles, image_shape):
#     """Remove circles whose boundary falls outside the image."""
#     h, w = image_shape[:2]
#     return [(x, y, r) for (x, y, r) in circles
#             if x - r >= 0 and y - r >= 0 and x + r < w and y + r < h]


# def scale_circles(circles, scale=0.5):
#     """Scale circle coords back after downscaling."""
#     factor = int(1 / scale)
#     return [(x * factor, y * factor, r * factor) for (x, y, r) in circles]


# def detect_and_validate(preprocessed_image, original_shape, params=None):
#     """Detect, filter, check overlap. Returns (circles, has_overlap)."""
#     circles     = detect_coins(preprocessed_image, params)
#     circles     = filter_circles(circles, original_shape)
#     has_overlap = check_overlap(circles) if len(circles) > 1 else False
#     return circles, has_overlap


# # ════════════════════════════════════════════════════════════
# #  STEP 4 — CROP ROI
# # ════════════════════════════════════════════════════════════

# def crop_roi(image, x, y, r, padding=5):
#     """Extract individual coin patch from the full image."""
#     h, w = image.shape[:2]
#     x1 = max(0, x - r - padding)
#     y1 = max(0, y - r - padding)
#     x2 = min(w, x + r + padding)
#     y2 = min(h, y + r + padding)
#     return image[y1:y2, x1:x2]


# # ════════════════════════════════════════════════════════════
# #  STEP 5 — MORPHOLOGICAL OPERATIONS
# # ════════════════════════════════════════════════════════════

# def clean_coin_mask(crop):
#     """
#     Clean coin crop with morphological ops.
#     Closing fills gaps, Opening removes noise.
#     Returns cleaned binary mask.
#     """
#     if crop is None or crop.size == 0:
#         return None
#     gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
#     _, binary = cv2.threshold(gray, 0, 255,
#                               cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
#     closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
#     opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN,  kernel, iterations=1)
#     return opened


# def watershed_separation(image, circles):
#     """Separate overlapping coins via watershed."""
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     _, binary = cv2.threshold(gray, 0, 255,
#                               cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
#     kernel   = np.ones((3, 3), np.uint8)
#     sure_bg  = cv2.dilate(binary, kernel, iterations=3)
#     dist     = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
#     _, sure_fg = cv2.threshold(dist, 0.5 * dist.max(), 255, 0)
#     sure_fg  = np.uint8(sure_fg)
#     unknown  = cv2.subtract(sure_bg, sure_fg)
#     _, markers = cv2.connectedComponents(sure_fg)
#     markers  = markers + 1
#     markers[unknown == 255] = 0
#     markers  = cv2.watershed(image, markers)
#     return markers


# def extract_coins_from_markers(image, markers):
#     """Extract (crop, x, y, r) per region after watershed."""
#     coins = []
#     for label in np.unique(markers):
#         if label <= 1:
#             continue
#         mask = np.zeros(markers.shape, dtype=np.uint8)
#         mask[markers == label] = 255
#         contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
#                                        cv2.CHAIN_APPROX_SIMPLE)
#         if not contours:
#             continue
#         cnt = max(contours, key=cv2.contourArea)
#         (cx, cy), radius = cv2.minEnclosingCircle(cnt)
#         cx, cy, radius = int(cx), int(cy), int(radius)
#         x1 = max(0, cx - radius)
#         y1 = max(0, cy - radius)
#         x2 = min(image.shape[1], cx + radius)
#         y2 = min(image.shape[0], cy + radius)
#         crop = image[y1:y2, x1:x2]
#         if crop.size > 0:
#             coins.append((crop, cx, cy, radius))
#     return coins


# # ════════════════════════════════════════════════════════════
# #  STEP 6 — FEATURE EXTRACTION
# # ════════════════════════════════════════════════════════════

# def extract_color_features(crop):
#     """
#     Mean HSV + normalized 3D HSV histogram (512-dim).

#     KEY feature for EGP coins is SATURATION:
#       1 EGP  → low-medium s  (silver ring dilutes saturation)
#       50 PT  → high s        (pure brass throughout)
#     """
#     if crop is None or crop.size == 0:
#         return 0.0, 0.0, 0.0, np.zeros(512)
#     resized = cv2.resize(crop, (64, 64), interpolation=cv2.INTER_AREA)
#     hsv     = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
#     mean_h, mean_s, mean_v, _ = cv2.mean(hsv)
#     hist = cv2.calcHist([hsv], [0, 1, 2], None,
#                         [8, 8, 8], [0, 180, 0, 256, 0, 256])
#     hist = cv2.normalize(hist, hist).flatten()
#     return mean_h, mean_s, mean_v, hist


# def extract_texture_features(crop):
#     """Laplacian variance — surface texture / engraving depth."""
#     if crop is None or crop.size == 0:
#         return 0.0
#     resized = cv2.resize(crop, (64, 64), interpolation=cv2.INTER_AREA)
#     gray    = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
#     return cv2.Laplacian(gray, cv2.CV_64F).var()


# def extract_bimetallic_ratio(crop):
#     """
#     Compute ratio of gold pixels to silver pixels inside the coin.

#     1 EGP  → ~equal gold + silver  → ratio ~0.8–1.5
#     50 PT  → almost all gold       → ratio  > 3.0

#     This is the most discriminative single feature for these two coins.
#     """
#     if crop is None or crop.size == 0:
#         return 0.0
#     resized = cv2.resize(crop, (64, 64), interpolation=cv2.INTER_AREA)
#     hsv     = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)

#     # circular mask — ignore background corners
#     mask = np.zeros((64, 64), dtype=np.uint8)
#     cv2.circle(mask, (32, 32), 30, 255, -1)

#     # gold/brass: warm hue, high saturation
#     gold_mask   = cv2.inRange(hsv, (10, 80, 80), (35, 255, 255))
#     gold_mask   = cv2.bitwise_and(gold_mask, mask)

#     # silver: low saturation, high brightness
#     silver_mask = cv2.inRange(hsv, (0, 0, 150), (180, 60, 255))
#     silver_mask = cv2.bitwise_and(silver_mask, mask)

#     gold_px   = cv2.countNonZero(gold_mask)
#     silver_px = cv2.countNonZero(silver_mask)

#     if silver_px == 0:
#         return float(gold_px)
#     return float(gold_px) / float(silver_px)


# def extract_features(crop, radius):
#     """
#     Full feature vector for one coin crop:
#     [radius, mean_H, mean_S, mean_V, hist×512, texture, bimetallic_ratio]
#     → 518 values total
#     """
#     mean_h, mean_s, mean_v, hist = extract_color_features(crop)
#     texture    = extract_texture_features(crop)
#     bimetallic = extract_bimetallic_ratio(crop)
#     return np.concatenate([
#         [float(radius)],
#         [mean_h, mean_s, mean_v],
#         hist.astype(np.float32),
#         [texture],
#         [bimetallic],
#     ]).astype(np.float32)


# # ════════════════════════════════════════════════════════════
# #  STEP 7 — CLASSIFICATION (VOTING)
# # ════════════════════════════════════════════════════════════

# def load_templates(templates_dir=TEMPLATES_DIR):
#     """Load reference templates from data/templates/<label>/*.jpg"""
#     templates = {}
#     if not os.path.exists(templates_dir):
#         print(f"[WARN] Templates folder not found: {templates_dir}")
#         return templates
#     for label in os.listdir(templates_dir):
#         label_dir = os.path.join(templates_dir, label)
#         if not os.path.isdir(label_dir):
#             continue
#         images = []
#         for fname in sorted(os.listdir(label_dir)):
#             if fname.lower().endswith((".jpg", ".jpeg", ".png")):
#                 img = cv2.imread(os.path.join(label_dir, fname))
#                 if img is not None:
#                     images.append(img)
#         if images:
#             templates[label] = images
#             print(f"[INFO] Loaded {len(images)} templates for {label}")
#     return templates


# def size_vote(radius):
#     """1 EGP (52px) vs 50 PT (45px) — pick closest."""
#     return min(COIN_SIZES, key=lambda k: abs(COIN_SIZES[k] - radius))


# def color_vote(crop):
#     """
#     Saturation threshold — strongest single signal for these two coins.
#     mean_S < 140  → bimetallic → EGP_1
#     mean_S >= 140 → brass      → EGP_0.50
#     """
#     _, mean_s, _, _ = extract_color_features(crop)
#     return "EGP_1" if mean_s < 140 else "EGP_0.50"


# def bimetallic_vote(crop):
#     """
#     Gold/silver pixel ratio.
#     1 EGP  → ratio < 2.5  (has a silver ring)
#     50 PT  → ratio > 2.5  (all gold, no ring)
#     """
#     ratio = extract_bimetallic_ratio(crop)
#     return "EGP_1" if ratio < 2.5 else "EGP_0.50"


# def template_vote(crop, templates):
#     """Normalized cross-correlation against stored templates."""
#     if not templates:
#         return "unknown"
#     best_score = -1
#     best_label = "unknown"
#     for label, tmpl_list in templates.items():
#         for tmpl in tmpl_list:
#             try:
#                 resized_tmpl = cv2.resize(tmpl, (crop.shape[1], crop.shape[0]),
#                                           interpolation=cv2.INTER_AREA)
#                 result = cv2.matchTemplate(crop, resized_tmpl,
#                                            cv2.TM_CCOEFF_NORMED)
#                 score  = float(result.max())
#                 if score > best_score:
#                     best_score = score
#                     best_label = label
#             except cv2.error:
#                 continue
#     return best_label


# def classify_coin(crop, radius, templates, use_svm=False):
#     """
#     Classify one coin via majority voting:
#       Vote 1 — size          (radius)
#       Vote 2 — color         (saturation threshold)
#       Vote 3 — bimetallic    (gold/silver ratio)
#       Vote 4 — template      (if templates available)

#     Returns (label, confidence).
#     """
#     if use_svm and os.path.exists(MODEL_PATH):
#         return svm_predict(crop, radius)

#     if crop is None or crop.size == 0:
#         return "unknown", 0.0

#     votes = [size_vote(radius), color_vote(crop), bimetallic_vote(crop)]
#     if templates:
#         votes.append(template_vote(crop, templates))

#     label      = max(set(votes), key=votes.count)
#     confidence = votes.count(label) / len(votes)

#     if confidence < 0.4:
#         label = "unknown"

#     return label, confidence


# # ════════════════════════════════════════════════════════════
# #  STEP 8 — DENOMINATION LOOKUP + SUM
# # ════════════════════════════════════════════════════════════

# def compute_total(labels):
#     return sum(COIN_VALUES.get(label, 0.0) for label in labels)


# # ════════════════════════════════════════════════════════════
# #  STEP 9 — DRAW OVERLAY
# # ════════════════════════════════════════════════════════════

# def draw_overlay(frame, circles, labels, confidences, total):
#     for (x, y, r), label, conf in zip(circles, labels, confidences):
#         color   = (0, 220, 80) if label != "unknown" else (0, 0, 220)
#         display = COIN_LABELS.get(label, label)
#         cv2.circle(frame, (x, y), r, color, 2)
#         cv2.circle(frame, (x, y), 3, color, -1)
#         cv2.putText(frame, f"{display} {conf:.0%}",
#                     (x - 32, y - r - 8),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
#     cv2.rectangle(frame, (10, 10), (290, 58), (0, 0, 0), -1)
#     cv2.putText(frame, f"Total: {total:.2f} EGP",
#                 (18, 44), cv2.FONT_HERSHEY_SIMPLEX,
#                 1.1, (0, 220, 255), 2, cv2.LINE_AA)
#     return frame


# # ════════════════════════════════════════════════════════════
# #  SVM CLASSIFIER — Phase 2
# # ════════════════════════════════════════════════════════════

# # def svm_train(data_dir=DATASET_PATH):
# #     """
# #     Train SVM on labeled images from your dataset.

# #     Your dataset structure:
# #       /home/adham/Documents/Coin-Counting-System/EGP Coin Dataset/
# #           batch1/  ← 1 EGP  (coin1.jpeg, coin2.jpeg ...)
# #           batch2/  ← 50 PT  (coin1.jpeg, coin2.jpeg ...)

# #     If your folder names differ, update DATASET_LABEL_MAP at the top.
# #     """
# #     try:
# #         from sklearn.svm import SVC
# #         from sklearn.preprocessing import StandardScaler
# #         from sklearn.model_selection import train_test_split
# #         from sklearn.metrics import classification_report
# #         import joblib
# #     except ImportError:
# #         print("[ERROR] Run: pip install scikit-learn joblib")
# #         return

# #     print(f"\n[INFO] Loading dataset from:\n  {data_dir}\n")

# #     if not os.path.exists(data_dir):
# #         print(f"[ERROR] Dataset not found: {data_dir}")
# #         return

# #     X, y        = [], []
# #     label_counts = {}

# #     for root, dirs, files in os.walk(data_dir):
# #         folder = (os.path.basename(root)
# #                   .lower().strip()
# #                   .replace(" ", "").replace("-", ""))
# #         label  = DATASET_LABEL_MAP.get(folder)
# #         if label is None:
# #             continue
# #         for fname in files:
# #             if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
# #                 continue
# #             img = cv2.imread(os.path.join(root, fname))
# #             if img is None:
# #                 continue
# #             radius   = min(img.shape[:2]) // 2
# #             features = extract_features(img, radius)
# #             X.append(features)
# #             y.append(label)
# #             label_counts[label] = label_counts.get(label, 0) + 1

# #     if not X:
# #         print("[ERROR] No images loaded.")
# #         print("  Folders found in dataset:")
# #         for root, dirs, _ in os.walk(data_dir):
# #             for d in dirs:
# #                 print(f"    {d}")
# #         print("\n  Update DATASET_LABEL_MAP to match your folder names.")
# #         return

# #     print("[INFO] Images loaded:")
# #     for label, count in label_counts.items():
# #         print(f"  {COIN_LABELS[label]:10s} → {count} images")

# #     X = np.array(X)
# #     y = np.array(y)

# #     X_train, X_test, y_train, y_test = train_test_split(
# #         X, y, test_size=0.2, random_state=42, stratify=y
# #     )

# #     scaler  = StandardScaler()
# #     X_train = scaler.fit_transform(X_train)
# #     X_test  = scaler.transform(X_test)

# #     print("\n[INFO] Training SVM (rbf kernel) ...")
# #     clf = SVC(kernel="rbf", C=10, gamma="scale", probability=True)
# #     clf.fit(X_train, y_train)

# #     y_pred = clf.predict(X_test)
# #     print("\n── Classification Report ──────────────────────────")
# #     print(classification_report(
# #         y_test, y_pred,
# #         target_names=[COIN_LABELS[l] for l in sorted(set(y))]
# #     ))

# #     os.makedirs(MODELS_DIR, exist_ok=True)
# #     joblib.dump((clf, scaler), MODEL_PATH)
# #     print(f"[INFO] Model saved → {MODEL_PATH}")


# # def svm_predict(crop, radius):
# #     """Predict using saved SVM. Falls back on failure."""
# #     try:
# #         import joblib
# #         clf, scaler = joblib.load(MODEL_PATH)
# #         features    = extract_features(crop, radius).reshape(1, -1)
# #         features_sc = scaler.transform(features)
# #         label       = clf.predict(features_sc)[0]
# #         confidence  = clf.predict_proba(features_sc).max()
# #         return label, float(confidence)
# #     except Exception as e:
# #         print(f"[WARN] SVM failed ({e}), falling back to voting")
# #         return "unknown", 0.0


# # ════════════════════════════════════════════════════════════
# #  CALIBRATION
# # ════════════════════════════════════════════════════════════

# def run_calibration():
#     """
#     Live webcam calibration.
#     Place each coin in frame → press SPACE → measure radius + HSV.
#     Prints values to paste into COIN_SIZES and COIN_COLORS above.
#     Press Q to quit.
#     """
#     print("\n── Calibration Mode ────────────────────────────────")
#     print("  1. Place ONE coin flat in front of camera")
#     print("  2. Press SPACE to capture its measurements")
#     print("  3. Repeat for both denominations")
#     print("  4. Press Q when done — paste printed values into main.py")
#     print("────────────────────────────────────────────────────\n")

#     cap = cv2.VideoCapture(0)
#     if not cap.isOpened():
#         print("[ERROR] Cannot open webcam.")
#         return

#     order    = ["1 EGP (1 Pound)", "50 PT (50 Piastres)"]
#     keys     = ["EGP_1", "EGP_0.50"]
#     measured = {}
#     idx      = 0

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         preprocessed = preprocess(frame)
#         circles      = detect_coins(preprocessed)
#         circles      = filter_circles(circles, frame.shape)

#         display = frame.copy()
#         for (x, y, r) in circles:
#             cv2.circle(display, (x, y), r, (0, 220, 80), 2)

#         txt = order[idx] if idx < len(order) else "Done — press Q"
#         cv2.putText(display, f"Place: {txt}",
#                     (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 255), 2)
#         cv2.putText(display, "SPACE = capture    Q = quit",
#                     (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

#         cv2.imshow("Calibration", display)
#         key = cv2.waitKey(1) & 0xFF

#         if key == ord('q') or idx >= len(keys):
#             break

#         if key == ord(' ') and circles:
#             x, y, r = max(circles, key=lambda c: c[2])
#             crop    = crop_roi(frame, x, y, r)
#             mean_h, mean_s, mean_v, _ = extract_color_features(crop)
#             bim     = extract_bimetallic_ratio(crop)

#             measured[keys[idx]] = {
#                 "radius": r,
#                 "h": round(mean_h), "s": round(mean_s), "v": round(mean_v),
#                 "bimetallic_ratio": round(bim, 2),
#             }
#             print(f"[CAPTURED] {order[idx]}")
#             print(f"  radius           : {r} px")
#             print(f"  H={mean_h:.0f}  S={mean_s:.0f}  V={mean_v:.0f}")
#             print(f"  bimetallic_ratio : {bim:.2f}\n")
#             idx += 1

#     cap.release()
#     cv2.destroyAllWindows()

#     if measured:
#         print("\n── Paste into main.py ──────────────────────────────")
#         print("\nCOIN_SIZES = {")
#         for k, v in measured.items():
#             print(f'    "{k}": {v["radius"]},')
#         print("}")
#         print("\nCOIN_COLORS = {")
#         for k, v in measured.items():
#             print(f'    "{k}": {{"h": {v["h"]}, "s": {v["s"]}, "v": {v["v"]}}},')
#         print("}")
#         with open("calibration_results.json", "w") as f:
#             json.dump(measured, f, indent=2)
#         print("\n  Also saved to calibration_results.json")


# # ════════════════════════════════════════════════════════════
# #  PIPELINE — SINGLE IMAGE
# # ════════════════════════════════════════════════════════════

# def run_image(image_path, use_svm=False, show=True):
#     """Run full 9-step pipeline on a single image."""
#     if not os.path.exists(image_path):
#         print(f"[ERROR] Image not found: {image_path}")
#         return
#     image = cv2.imread(image_path)
#     if image is None:
#         print(f"[ERROR] Could not read: {image_path}")
#         return

#     templates = load_templates()
#     os.makedirs(OUTPUTS_DIR, exist_ok=True)
#     print(f"\n[INFO] Processing: {image_path}")

#     preprocessed         = preprocess(image)
#     circles, has_overlap = detect_and_validate(preprocessed, image.shape)
#     print(f"[INFO] {len(circles)} coin(s) detected | overlap={has_overlap}")

#     if has_overlap and len(circles) > 1:
#         print("[INFO] Applying watershed separation...")
#         markers  = watershed_separation(image, circles)
#         ws_coins = extract_coins_from_markers(image, markers)
#         if ws_coins:
#             circles = [(cx, cy, r) for (_, cx, cy, r) in ws_coins]

#     labels, confidences = [], []
#     for (x, y, r) in circles:
#         crop       = crop_roi(image, x, y, r)
#         clean_coin_mask(crop)
#         label, conf = classify_coin(crop, r, templates, use_svm)
#         labels.append(label)
#         confidences.append(conf)
#         print(f"  ({x},{y}) r={r}px → {COIN_LABELS.get(label, label)} "
#               f"conf={conf:.0%}")

#     total    = compute_total(labels)
#     print(f"\n  ── Total: {total:.2f} EGP ──")

#     result   = draw_overlay(image.copy(), circles, labels, confidences, total)
#     out_path = os.path.join(OUTPUTS_DIR,
#                             f"result_{os.path.basename(image_path)}")
#     cv2.imwrite(out_path, result)
#     print(f"[INFO] Saved → {out_path}")

#     if show:
#         cv2.imshow("Coin Counter — Result", result)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()

#     return labels, total


# # ════════════════════════════════════════════════════════════
# #  PIPELINE — REAL-TIME WEBCAM
# # ════════════════════════════════════════════════════════════

# def run_realtime(use_svm=False):
#     """
#     Live coin detection. Q = quit | S = snapshot | C = clear total
#     """
#     print("\n── Real-Time Mode ──────────────────────────────────")
#     print(f"  Classifier : {'SVM' if use_svm and os.path.exists(MODEL_PATH) else 'Voting'}")
#     print("  Q = quit  |  S = snapshot  |  C = clear total\n")

#     cap = cv2.VideoCapture(0)
#     if not cap.isOpened():
#         print("[ERROR] Cannot open webcam.")
#         return

#     templates        = load_templates()
#     os.makedirs(OUTPUTS_DIR, exist_ok=True)

#     stability_buffer = deque(maxlen=5)
#     locked_total     = 0.0
#     locked_labels    = []
#     locked_conf      = []
#     locked_circles   = []
#     fps_timer        = time.time()
#     fps              = 0
#     frame_cnt        = 0

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         preprocessed = preprocess(frame, blur_kernel=(5, 5))
#         small        = resize_for_detection(preprocessed, scale=0.5)
#         circles_raw  = detect_coins(small)
#         circles      = scale_circles(circles_raw, scale=0.5)
#         circles      = filter_circles(circles, frame.shape)

#         labels, confidences = [], []
#         for (x, y, r) in circles:
#             crop       = crop_roi(frame, x, y, r)
#             clean_coin_mask(crop)
#             label, conf = classify_coin(crop, r, templates, use_svm)
#             labels.append(label)
#             confidences.append(conf)

#         stability_buffer.append(len(circles))
#         if (len(stability_buffer) == stability_buffer.maxlen and
#                 len(set(stability_buffer)) == 1):
#             locked_total   = compute_total(labels)
#             locked_labels  = labels
#             locked_conf    = confidences
#             locked_circles = circles

#         frame_cnt += 1
#         if time.time() - fps_timer >= 1.0:
#             fps       = frame_cnt
#             frame_cnt = 0
#             fps_timer = time.time()

#         display = draw_overlay(frame.copy(), locked_circles,
#                                locked_labels, locked_conf, locked_total)
#         cv2.putText(display, f"FPS: {fps}",
#                     (display.shape[1] - 100, 36),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (160, 160, 160), 1)
#         cv2.putText(display, f"Coins: {len(locked_circles)}",
#                     (display.shape[1] - 120, 62),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (160, 160, 160), 1)

#         cv2.imshow("Coin Counter — EGP Live", display)

#         key = cv2.waitKey(1) & 0xFF
#         if key == ord('q'):
#             break
#         elif key == ord('s'):
#             fname = os.path.join(OUTPUTS_DIR,
#                                  f"snapshot_{int(time.time())}.jpg")
#             cv2.imwrite(fname, display)
#             print(f"[INFO] Snapshot → {fname}")
#         elif key == ord('c'):
#             locked_total, locked_labels = 0.0, []
#             locked_conf, locked_circles = [], []
#             stability_buffer.clear()
#             print("[INFO] Total cleared")

#     cap.release()
#     cv2.destroyAllWindows()
#     print("[INFO] Session ended.")


# # ════════════════════════════════════════════════════════════
# #  ENTRY POINT
# # ════════════════════════════════════════════════════════════

# def main():
#     parser = argparse.ArgumentParser(
#         description="EGP Coin Counter (1 EGP | 50 PT) — CSE483"
#     )
#     parser.add_argument("--mode",
#                         choices=["calibrate", "image", "realtime", "train"],
#                         default="realtime")
#     parser.add_argument("--input",  type=str, default=None,
#                         help="Image path (--mode image)")
#     parser.add_argument("--data",   type=str, default=DATASET_PATH,
#                         help=f"Dataset path for training (default: {DATASET_PATH})")
#     parser.add_argument("--svm",    action="store_true",
#                         help="Use SVM classifier")
#     parser.add_argument("--noshow", action="store_true",
#                         help="Skip result window in image mode")
#     args = parser.parse_args()

#     if args.mode == "calibrate":
#         run_calibration()
#     elif args.mode == "image":
#         if not args.input:
#             print("[ERROR] --input required.")
#             print("  Example: python main.py --mode image --input data/raw/test.jpg")
#             sys.exit(1)
#         run_image(args.input, use_svm=args.svm, show=not args.noshow)
#     elif args.mode == "realtime":
#         run_realtime(use_svm=args.svm)
#     elif args.mode == "train":
#         svm_train(data_dir=args.data)


# if __name__ == "__main__":
#     main()