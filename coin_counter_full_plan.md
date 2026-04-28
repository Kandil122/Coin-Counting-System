# Coin Counting System — Full Project Plan
**Course:** CSE483: Computer Vision — Ain Shams University  
**Approach:** Classical CV + optional SVM upgrade  
**Mode:** Static images → Real-time webcam  
**Currency:** Egyptian Pounds (EGP) only

---

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [Correct Pipeline Order](#2-correct-pipeline-order)
3. [Egyptian Coins Reference](#3-egyptian-coins-reference)
4. [Folder Structure](#4-folder-structure)
5. [Libraries & Requirements](#5-libraries--requirements)
6. [Phase 1 — Classical CV Pipeline](#6-phase-1--classical-cv-pipeline)
7. [Phase 2 — SVM Upgrade](#7-phase-2--svm-upgrade)
8. [Phase 3 — Real-Time Webcam](#8-phase-3--real-time-webcam)
9. [Notebooks Breakdown](#9-notebooks-breakdown)
10. [Scripts Breakdown](#10-scripts-breakdown)
11. [Calibration Guide](#11-calibration-guide)
12. [Known Challenges & Fixes](#12-known-challenges--fixes)
13. [Evaluation Metrics](#13-evaluation-metrics)
14. [Suggested Timeline](#14-suggested-timeline)

---

## 1. Project Overview

Detect Egyptian pound coins in an image or live webcam feed, classify each coin by denomination, and display the total value in EGP with an annotated overlay.

**Coins in scope:**

| Denomination | Value |
|---|---|
| 1 Pound | 1.00 EGP |
| 50 Piastres | 0.50 EGP |
| 25 Piastres | 0.25 EGP |

**Core idea:** No deep learning required. The system uses:
- **Preprocessing** to clean the image before detection
- **Circular Hough Transform** to detect coin locations and sizes
- **Morphological operations** to clean coin masks and handle overlaps
- **Feature extraction** (size, color, texture) per detected coin
- **Majority voting** to classify each coin into a denomination label
- **Optional SVM** as a drop-in upgrade for better accuracy on similar-looking denominations
- **OpenCV webcam loop** for real-time mode with a live overlay

---

## 2. Correct Pipeline Order

This is the confirmed, correct sequence. Order matters — each step depends on the one before it.

```
1. Capture image
        ↓
2. Preprocessing
   grayscale → Gaussian blur → contrast boost
        ↓
3. Hough Transform (detection)
   HoughCircles → list of (x, y, radius) per coin
        ↓
4. Crop each coin ROI
   extract individual coin patch from full image
        ↓
5. Morphological operations (per crop)
   clean coin mask, close gaps, separate overlapping regions
   → watershed fallback if coins overlap
        ↓
6. Feature extraction (per crop)
   size (radius), color (HSV histogram), texture (Laplacian variance)
        ↓
7. Classification
   voting (size + color + template) → OR → SVM predict
        ↓
8. Denomination lookup
   label → EGP value from coin_map.py
        ↓
9. Sum + display
   total EGP value drawn on image / live frame
```

> **Key clarifications:**
> - Morphological operations happen **after** detection and cropping — not before Hough Transform
> - Feature extraction happens **after** morphology — on the cleaned coin crop
> - Hough Transform runs on the preprocessed full image — not on individual crops

---

## 3. Egyptian Coins Reference

This section defines everything the system needs to know about each coin.
These values are populated by `calibration.py` and stored in `coin_map.py`.

### Physical properties (real-world)

| Coin | Diameter (mm) | Color | Material |
|---|---|---|---|
| 1 Pound | 23.0 mm | Gold / brass | Bimetallic |
| 50 Piastres | 21.0 mm | Silver | Nickel-plated steel |
| 25 Piastres | 18.0 mm | Silver / copper edge | Nickel-plated steel |

### Visual characteristics (for classification)

| Coin | HSV profile | Distinguishing feature |
|---|---|---|
| 1 Pound | High hue (~20–30), high saturation | Gold tone, largest diameter |
| 50 Piastres | Low saturation, high value (silver) | Medium diameter, fully silver |
| 25 Piastres | Low-medium saturation | Smallest diameter, silver-copper edge |

> **The hard problem:** 50 Piastres and 25 Piastres are both silver-toned and only ~3mm apart
> in diameter. Under poor lighting, size and color alone may not be enough — this is exactly
> where the SVM upgrade pays off.

### `coin_map.py` — EGP only

```python
# Sizes populated by calibration.py (values in pixels at your fixed camera height)
COIN_SIZES = {
    "EGP_1":    52,    # 1 Pound   — largest
    "EGP_0.50": 45,    # 50 Piastres — medium
    "EGP_0.25": 38,    # 25 Piastres — smallest
}

COIN_COLORS = {
    "EGP_1":    {"h": 25, "s": 180, "v": 160},   # gold/brass
    "EGP_0.50": {"h": 0,  "s": 20,  "v": 200},   # silver
    "EGP_0.25": {"h": 15, "s": 60,  "v": 180},   # silver with copper edge
}

COIN_VALUES = {
    "EGP_1":    1.00,
    "EGP_0.50": 0.50,
    "EGP_0.25": 0.25,
}
```

---

## 4. Folder Structure

```
coin_counter/
│
├── data/
│   ├── raw/                    # Original coin photos (varied lighting, angles)
│   ├── processed/              # Cropped ROIs extracted per detected coin
│   └── templates/              # Reference photos per denomination
│       ├── EGP_1/              # 12 rotations of the 1 Pound coin
│       ├── EGP_0.50/           # 12 rotations of the 50 Piastres coin
│       └── EGP_0.25/           # 12 rotations of the 25 Piastres coin
│
├── notebooks/
│   ├── 01_data_collection.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_detection.ipynb
│   ├── 04_morphology.ipynb
│   ├── 05_feature_extraction.ipynb
│   ├── 06_classification.ipynb
│   ├── 07_svm_training.ipynb
│   └── 08_evaluation.ipynb
│
├── src/
│   ├── __init__.py
│   ├── preprocess.py           # Grayscale, blur, contrast helpers
│   ├── detect.py               # HoughCircles + overlap check
│   ├── morphology.py           # Morphological ops + watershed overlap handler
│   ├── features.py             # Feature extraction (size, HSV, texture)
│   ├── classify.py             # Voting logic (size + color + template)
│   ├── svm_classifier.py       # Train / load / predict with scikit-learn SVM
│   ├── coin_map.py             # EGP denomination lookup (sizes, colors, values)
│   ├── pipeline.py             # Static image end-to-end run
│   ├── realtime.py             # Webcam loop, overlay, stability buffer
│   └── utils.py                # Shared helpers (crop ROI, draw, HSV extract)
│
├── models/
│   └── svm_coin_classifier.pkl # Saved trained SVM (joblib dump)
│
├── outputs/                    # Annotated result images + session log CSVs
│
├── calibration.py              # One-time run: measure radii + HSV, save to coin_map
├── requirements.txt
└── README.md
```

---

## 5. Libraries & Requirements

### `requirements.txt`
```
opencv-python>=4.8.0
numpy>=1.24.0
scikit-learn>=1.3.0
joblib>=1.3.0
matplotlib>=3.7.0
pandas>=2.0.0
jupyter>=1.0.0
```

### Install
```bash
pip install -r requirements.txt
```

### What each library does

| Library | Role in this project |
|---|---|
| `opencv-python` | Preprocessing, Hough Transform, morphology, template matching, webcam, overlay |
| `numpy` | Array operations, image slicing, feature vectors, distance math |
| `scikit-learn` | SVM training and prediction (Phase 2 only) |
| `joblib` | Save and load trained SVM model to disk |
| `matplotlib` | Visualize results in notebooks, plot confusion matrix |
| `pandas` | Log session results to CSV, build labeled dataset for SVM |

> No PyTorch, TensorFlow, GPU, or deep learning of any kind required.
> Runs fully on CPU on a standard laptop.

---

## 6. Phase 1 — Classical CV Pipeline

### Step 1 — Calibration (run once before everything)

Place each coin individually under the camera at your fixed shooting height.
Run `calibration.py` — it detects the coin, measures its radius in pixels and mean HSV,
and saves the values into `coin_map.py`.

**Calibration requirements:**
- Fixed camera height — do not move it after calibration
- Flat, matte surface (plain white or dark paper)
- Even diffuse lighting — no direct sunlight or single-point lamps
- One coin at a time, flat on the surface

---

### Step 2 — Preprocessing (`preprocess.py`)

Runs on the full image before any detection.

```python
import cv2

def preprocess(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    # CLAHE improves contrast on reflective Egyptian coin surfaces
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(blurred)
    return enhanced

def resize_for_detection(image, scale=0.5):
    return cv2.resize(image, (0, 0), fx=scale, fy=scale)
```

---

### Step 3 — Hough Transform / Detection (`detect.py`)

Runs on the preprocessed full image. Returns circle coordinates for each detected coin.

```python
import cv2
import numpy as np

def detect_coins(preprocessed_image):
    circles = cv2.HoughCircles(
        preprocessed_image,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=40,       # minimum distance between coin centers
        param1=100,       # Canny edge upper threshold
        param2=30,        # accumulator threshold — lower = more detections
        minRadius=20,     # smallest coin (25 Piastres)
        maxRadius=80      # largest coin (1 Pound)
    )
    if circles is None:
        return []
    return np.round(circles[0, :]).astype("int")

def check_overlap(circles):
    for i in range(len(circles)):
        for j in range(i + 1, len(circles)):
            x1, y1, r1 = circles[i]
            x2, y2, r2 = circles[j]
            dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            if dist < r1 + r2:
                return True
    return False
```

---

### Step 4 — Crop Each Coin ROI (`utils.py`)

```python
import cv2

def crop_roi(image, x, y, r, padding=5):
    h, w = image.shape[:2]
    x1 = max(0, x - r - padding)
    y1 = max(0, y - r - padding)
    x2 = min(w, x + r + padding)
    y2 = min(h, y + r + padding)
    return image[y1:y2, x1:x2]
```

---

### Step 5 — Morphological Operations (`morphology.py`)

Runs per coin crop after detection. Cleans the mask and handles overlapping coins.

```python
import cv2
import numpy as np

def clean_coin_mask(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=1)
    return opened

def watershed_separation(image, circles):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    sure_bg = cv2.dilate(binary, kernel, iterations=3)
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist, 0.5 * dist.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(image, markers)
    return markers
```

---

### Step 6 — Feature Extraction (`features.py`)

Runs on each cleaned coin crop. Produces the feature vector used for classification.

```python
import cv2
import numpy as np

def extract_color_features(crop):
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mean_h, mean_s, mean_v = cv2.mean(hsv)[:3]
    hist = cv2.calcHist([hsv], [0, 1, 2], None,
                        [8, 8, 8], [0, 180, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return mean_h, mean_s, mean_v, hist

def extract_texture_features(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def extract_features(crop, radius):
    mean_h, mean_s, mean_v, hist = extract_color_features(crop)
    texture = extract_texture_features(crop)
    return np.concatenate([
        [radius],
        [mean_h, mean_s, mean_v],
        hist,
        [texture]
    ])
```

---

### Step 7 — Classification (`classify.py`)

Three independent votes combined into one majority decision.

```python
import cv2
import numpy as np
from coin_map import COIN_SIZES, COIN_COLORS

def size_vote(radius):
    return min(COIN_SIZES, key=lambda k: abs(COIN_SIZES[k] - radius))

def color_vote(crop):
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mean_hsv = cv2.mean(hsv)[:3]
    return min(COIN_COLORS, key=lambda k: np.linalg.norm(
        np.array(list(COIN_COLORS[k].values())) - np.array(mean_hsv)
    ))

def template_vote(crop, templates):
    best_score, best_label = -1, "unknown"
    for label, template in templates.items():
        resized = cv2.resize(template, (crop.shape[1], crop.shape[0]))
        result = cv2.matchTemplate(crop, resized, cv2.TM_CCOEFF_NORMED)
        score = result.max()
        if score > best_score:
            best_score, best_label = score, label
    return best_label

def classify_coin(crop, radius, templates):
    votes = [
        size_vote(radius),
        color_vote(crop),
        template_vote(crop, templates)
    ]
    label = max(set(votes), key=votes.count)
    confidence = votes.count(label) / 3
    if confidence < 0.5:
        label = "unknown"
    return label, confidence
```

---

### Step 8 — Denomination Lookup + Sum

```python
from coin_map import COIN_VALUES

def compute_total(labels):
    return sum(COIN_VALUES.get(label, 0) for label in labels)
```

---

### Step 9 — Output (`utils.py`)

```python
import cv2

def draw_overlay(frame, circles, labels, confidences, total):
    for (x, y, r), label, conf in zip(circles, labels, confidences):
        color = (0, 255, 100) if label != "unknown" else (0, 0, 255)
        cv2.circle(frame, (x, y), r, color, 2)
        cv2.putText(frame, f"{label} ({conf:.0%})",
                    (x - 30, y - r - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    cv2.putText(frame, f"Total: {total:.2f} EGP",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 200, 255), 2)
    return frame
```

---

## 7. Phase 2 — SVM Upgrade

Only do this if Phase 1 voting accuracy is below ~90%, which is most likely to happen
when the system confuses **50 Piastres and 25 Piastres** — they are both silver-toned
and only ~3mm apart in diameter.

### Data collection for SVM

| Coin | Target images |
|---|---|
| 1 Pound (EGP_1) | ~100 photos |
| 50 Piastres (EGP_0.50) | ~100 photos |
| 25 Piastres (EGP_0.25) | ~100 photos |
| **Total** | ~300 images |

Photograph under at least 3 different lighting conditions per denomination.

### Training (`07_svm_training.ipynb`)

```python
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

clf = SVC(kernel='rbf', C=10, gamma='scale', probability=True)
clf.fit(X_train, y_train)

print(classification_report(y_test, clf.predict(X_test)))
joblib.dump((clf, scaler), 'models/svm_coin_classifier.pkl')
```

### Drop-in replacement (`svm_classifier.py`)

```python
import joblib
from features import extract_features

def classify_coin(crop, radius, templates=None):
    clf, scaler = joblib.load('models/svm_coin_classifier.pkl')
    features = extract_features(crop, radius)
    features_scaled = scaler.transform([features])
    label = clf.predict(features_scaled)[0]
    confidence = clf.predict_proba(features_scaled).max()
    return label, confidence
```

Switch between voting and SVM with one import line in `pipeline.py` / `realtime.py`:

```python
# from classify import classify_coin       # Phase 1 — voting
from svm_classifier import classify_coin   # Phase 2 — SVM
```

---

## 8. Phase 3 — Real-Time Webcam

### Key modifications from static pipeline

| Aspect | Static | Real-time |
|---|---|---|
| Input | Single image file | `cv2.VideoCapture(0)` frame loop |
| Processing budget | Unlimited | ~33ms per frame (30fps) |
| Detection speed | Full resolution | Downscale to 50%, upscale coords back |
| Output | Saved image file | `cv2.imshow()` live overlay |
| Total display | Print to console | Drawn on frame with `cv2.putText()` |
| Stability | N/A | Lock total after N=5 consistent frames |
| Flickering | N/A | Rolling window smoothing |

### Core loop (`realtime.py`)

```python
import cv2
import time
from collections import deque
from preprocess import preprocess, resize_for_detection
from detect import detect_coins, check_overlap
from morphology import clean_coin_mask, watershed_separation
from features import extract_features
from svm_classifier import classify_coin   # or: from classify import classify_coin
from coin_map import COIN_VALUES
from utils import crop_roi, draw_overlay

cap = cv2.VideoCapture(0)
stability_buffer = deque(maxlen=5)
locked_total = 0.0
locked_labels = []
templates = {}  # load your templates dict here

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # step 2: preprocess
    preprocessed = preprocess(frame)

    # step 3: detect on downscaled frame for speed
    small = resize_for_detection(preprocessed, scale=0.5)
    circles_small = detect_coins(small)
    circles = [(x*2, y*2, r*2) for (x, y, r) in circles_small]

    # step 4-7: per coin
    labels, confidences = [], []
    for (x, y, r) in circles:
        crop = crop_roi(frame, x, y, r)
        if crop.size == 0:
            continue
        clean_coin_mask(crop)                        # step 5: morphology
        features = extract_features(crop, r)         # step 6: features
        label, conf = classify_coin(crop, r, templates)  # step 7: classify
        labels.append(label)
        confidences.append(conf)

    # stability check — update total only when count is consistent
    stability_buffer.append(len(labels))
    if len(set(stability_buffer)) == 1:
        locked_total = sum(COIN_VALUES.get(l, 0) for l in labels)
        locked_labels = labels

    # step 9: draw overlay
    frame = draw_overlay(frame, circles, locked_labels, confidences, locked_total)
    cv2.imshow("Coin Counter — EGP", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        filename = f"outputs/snapshot_{int(time.time())}.jpg"
        cv2.imwrite(filename, frame)
        print(f"Saved: {filename}")

cap.release()
cv2.destroyAllWindows()
```

---

## 9. Notebooks Breakdown

### `01_data_collection.ipynb`
- Photograph each of the 3 EGP denominations under 3+ lighting conditions
- Capture ~100 images per denomination for SVM dataset
- Build `data/templates/EGP_1/`, `EGP_0.50/`, `EGP_0.25/` — 12 rotations each (every 30°)
- Run `calibration.py` interactively, verify saved radius and HSV values

### `02_preprocessing.ipynb`
- Experiment with blur kernel sizes (3×3, 5×5, 9×9)
- Compare plain grayscale vs CLAHE on reflective Egyptian coin surfaces
- Visualize effect of each step on all 3 coin types
- Save best parameters as constants for `preprocess.py`

### `03_detection.ipynb`
- Tune all `HoughCircles` parameters with interactive sliders
- Test on images with 1, 3, and 6 coins mixed
- Test overlapping coins — especially 50 and 25 Piastres which are similarly sized
- Visualize pre/post watershed separation
- Document final working parameter values

### `04_morphology.ipynb`
- Apply closing, opening, dilation, erosion to EGP coin crops
- Visualize which operations best clean up the gold (1 Pound) vs silver (0.50, 0.25) masks
- Test watershed on intentionally overlapping pairs
- Document best kernel size and iteration count

### `05_feature_extraction.ipynb`
- Plot HSV histograms for all 3 denominations — confirm they are visually separable
- Plot radius distribution per denomination from your calibration photos
- Show Laplacian variance differences between coin types
- This notebook confirms your features are good enough before classification

### `06_classification.ipynb`
- Test size vote alone: how often does it separate 0.50 from 0.25?
- Test color vote alone: does gold/silver distinction work reliably?
- Test template matching alone: accuracy across 12 rotations
- Combine: majority vote accuracy on your full test set
- Identify which pairs fail most — this is your motivation for Phase 2 SVM

### `07_svm_training.ipynb`
- Load ~300 labeled coin images (100 per denomination)
- Extract feature vectors for all images
- Split train/test 80/20
- Train SVM, print `classification_report` per class
- Compare accuracy to Phase 1 voting — show the improvement
- Save model to `models/svm_coin_classifier.pkl`

### `08_evaluation.ipynb`
- Side-by-side: voting accuracy vs SVM accuracy per denomination
- Confusion matrix heatmap for both approaches
- Per-class precision, recall, F1
- Failure analysis: are 0.50 and 0.25 still confused? Under what conditions?
- Final summary table for your written report

---

## 10. Scripts Breakdown

### `calibration.py`
Run once before everything. Opens camera, user places one coin at a time and presses
spacebar. Script detects the coin, measures radius and HSV, saves to `coin_map.py`.

### `preprocess.py`
- `preprocess(image)` → grayscale + Gaussian blur + CLAHE → cleaned image
- `resize_for_detection(image, scale=0.5)` → downscale for real-time speed

### `detect.py`
- `detect_coins(preprocessed)` → list of `(x, y, r)`
- `check_overlap(circles)` → True if any two circles intersect

### `morphology.py`
- `clean_coin_mask(crop)` → closing + opening on binary coin mask
- `watershed_separation(image, circles)` → separates overlapping coins

### `features.py`
- `extract_color_features(crop)` → mean HSV + histogram
- `extract_texture_features(crop)` → Laplacian variance
- `extract_features(crop, radius)` → full combined feature vector

### `classify.py`
- `size_vote(radius)` → label string
- `color_vote(crop)` → label string
- `template_vote(crop, templates)` → label string
- `classify_coin(crop, radius, templates)` → `(label, confidence)`

### `svm_classifier.py`
- `classify_coin(crop, radius, templates=None)` → `(label, confidence)`
- Identical interface to `classify.py` — one import line to swap

### `coin_map.py`
Three EGP-only dicts: `COIN_SIZES`, `COIN_COLORS`, `COIN_VALUES`.
Values populated by `calibration.py` and reflect your specific camera setup.

### `pipeline.py`
Accepts an image path, runs all 9 pipeline steps, saves annotated image to `outputs/`.

### `realtime.py`
Full webcam loop. Press `q` to quit, `s` to save snapshot to `outputs/`.
Stability buffer of 5 frames before locking displayed total.

### `utils.py`
- `crop_roi(image, x, y, r)` → padded square crop around circle
- `draw_overlay(frame, circles, labels, confidences, total)` → all annotations
- `extract_hsv_mean(crop)` → returns mean H, S, V

---

## 11. Calibration Guide

The most important step — wrong calibration means wrong results everywhere.

**Setup requirements:**
- Camera at its final, fixed shooting position
- Flat matte surface — plain white paper or dark felt works best
- Diffuse, even lighting — a desk lamp pointing at the ceiling is better than direct
- One coin at a time, flat, no shadows

**Steps:**
1. Set camera at its final position
2. Run `python calibration.py`
3. Place each denomination in frame, press spacebar
4. Repeat for all 3 denominations (1 Pound, 50 Piastres, 25 Piastres)
5. Verify `coin_map.py` values look reasonable
6. Run `python src/pipeline.py` on 3 test photos to confirm

**If results drift later:** Recalibrate under the same lighting you will demo under.

---

## 12. Known Challenges & Fixes

| Challenge | Why it happens | Fix |
|---|---|---|
| Coins not detected | `param2` too high | Lower `param2` to 25–30 |
| False circles detected | Background texture | Raise `param2`, use plain background |
| 50 and 25 Piastres merged | Coins touching, HoughCircles sees one | Watershed fallback in `morphology.py` |
| 50 and 25 Piastres confused | Similar size and color | Add SVM (Phase 2) |
| Template matching fails | Coin is rotated | Store 12 rotations per denomination |
| Total flickers in real-time | Frame-to-frame variation | Stability buffer N=5 frames |
| Slow real-time performance | Full-res HoughCircles is slow | Downscale frame to 50% before detection |
| Glare on 1 Pound gold surface | Reflective brass | Use diffuse/indirect lighting |
| Camera moved between runs | Pixel radius changes | Recalibrate at new position |
| Unknown label returned | All three votes disagree | Draw `?`, log to CSV for review |

---

## 13. Evaluation Metrics

### Phase 1 — voting
- Per-denomination accuracy (especially 50 vs 25 Piastres)
- Overall accuracy across all 3 coin types
- Confusion matrix — which pairs are most confused

### Phase 2 — SVM
- Same metrics directly compared to Phase 1
- Precision, recall, F1 per denomination (`classification_report`)
- Training vs test accuracy (check for overfitting on small ~300 image dataset)

### Phase 3 — real-time
- Average FPS (`cv2.getTickCount()` before and after each loop iteration)
- Stability latency: how many frames until total locks in
- Qualitative: does it hold under different lighting conditions

### For your written report
Show a comparison table: voting accuracy vs SVM accuracy per denomination.
Include confusion matrices for both. Annotate sample output images.
Explain the 50 vs 25 Piastres problem clearly — it is your main technical challenge.



*Generated for CSE483: Computer Vision — Ain Shams University*