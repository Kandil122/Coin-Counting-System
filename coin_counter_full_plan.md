# Coin Counting System — Full Project Plan
**Course:** CSE483: Computer Vision — Ain Shams University  
**Approach:** Classical CV + optional SVM upgrade  
**Mode:** Static images → Real-time webcam  
**Currencies:** Mixed (EGP, USD, EUR)

---

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [Correct Pipeline Order](#2-correct-pipeline-order)
3. [Folder Structure](#3-folder-structure)
4. [Libraries & Requirements](#4-libraries--requirements)
5. [Phase 1 — Classical CV Pipeline](#5-phase-1--classical-cv-pipeline)
6. [Phase 2 — SVM Upgrade](#6-phase-2--svm-upgrade)
7. [Phase 3 — Real-Time Webcam](#7-phase-3--real-time-webcam)
8. [Notebooks Breakdown](#8-notebooks-breakdown)
9. [Scripts Breakdown](#9-scripts-breakdown)
10. [Calibration Guide](#10-calibration-guide)
11. [Known Challenges & Fixes](#11-known-challenges--fixes)
12. [Evaluation Metrics](#12-evaluation-metrics)
13. [Suggested Timeline](#13-suggested-timeline)

---

## 1. Project Overview

Detect coins in an image or live webcam feed, classify each coin by denomination across multiple currencies, and display the total value with an annotated overlay.

**Core idea:** No deep learning required. The system uses:
- **Preprocessing** to clean the image before detection
- **Circular Hough Transform** to detect coin locations and sizes
- **Morphological operations** to clean coin masks and handle overlaps
- **Feature extraction** (size, color, texture) per detected coin
- **Majority voting** to classify each coin into a denomination label
- **Optional SVM** as a drop-in upgrade for better accuracy on mixed currencies
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
   size (radius), color (HSV histogram), texture (LBP / Gabor)
        ↓
7. Classification
   voting (size + color + template) → OR → SVM predict
        ↓
8. Denomination lookup
   label → value from coin_map.py
        ↓
9. Sum + display
   total value drawn on image / live frame
```

> **Key clarification:**
> - Morphological operations happen **after** detection and cropping — not before Hough Transform
> - Feature extraction happens **after** morphology — on the cleaned coin crop
> - Hough Transform runs on the preprocessed full image — not on crops

---

## 3. Folder Structure

```
coin_counter/
│
├── data/
│   ├── raw/                    # Original coin photos (varied lighting, angles)
│   ├── processed/              # Cropped ROIs extracted per detected coin
│   └── templates/              # 1 clean reference photo × 12 rotations per coin type
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
│   ├── preprocess.py           # Grayscale, blur, contrast helpers
│   ├── detect.py               # HoughCircles + overlap check
│   ├── morphology.py           # Morphological ops + watershed overlap handler
│   ├── features.py             # Feature extraction (size, HSV, texture)
│   ├── classify.py             # Voting logic (size + color + template)
│   ├── svm_classifier.py       # Train / load / predict with scikit-learn SVM
│   ├── coin_map.py             # Denomination lookup dict (EGP, USD, EUR)
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

## 4. Libraries & Requirements

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

## 5. Phase 1 — Classical CV Pipeline

### Step 1 — Calibration (run once before everything)

Place each coin individually under the camera at your fixed shooting height.
Run `calibration.py` — it detects the coin, measures its radius in pixels and mean HSV, and saves values into `coin_map.py`.

```python
# coin_map.py — auto-populated by calibration.py
COIN_SIZES = {
    "EGP_1":       52,   # radius in pixels at your fixed camera height
    "EGP_0.50":    45,
    "EGP_0.25":    38,
    "USD_quarter":  47,
    "USD_dime":     38,
    "EUR_1":        54,
}

COIN_COLORS = {
    "EGP_1":       {"h": 25, "s": 180, "v": 160},   # gold/brass
    "EGP_0.50":    {"h": 0,  "s": 20,  "v": 200},   # silver
    "EGP_0.25":    {"h": 15, "s": 60,  "v": 180},   # silver-copper
    "USD_quarter":  {"h": 0,  "s": 15,  "v": 195},
    "USD_dime":     {"h": 0,  "s": 12,  "v": 210},
    "EUR_1":        {"h": 22, "s": 160, "v": 155},
}

COIN_VALUES = {
    "EGP_1":       1.00,
    "EGP_0.50":    0.50,
    "EGP_0.25":    0.25,
    "USD_quarter":  0.25,
    "USD_dime":     0.10,
    "EUR_1":        1.00,
}
```

---

### Step 2 — Preprocessing (`preprocess.py`)

Runs on the full image before any detection.

```python
import cv2

def preprocess(image):
    # convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # blur to suppress noise before Hough
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    # optional: CLAHE for reflective coins under uneven lighting
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(blurred)
    return enhanced

def resize_for_detection(image, scale=0.5):
    # downscale for real-time speed
    return cv2.resize(image, (0, 0), fx=scale, fy=scale)
```

---

### Step 3 — Hough Transform / Detection (`detect.py`)

Runs on the preprocessed full image.

```python
import cv2
import numpy as np

def detect_coins(preprocessed_image):
    circles = cv2.HoughCircles(
        preprocessed_image,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=40,       # minimum distance between coin centers
        param1=100,       # Canny edge threshold
        param2=30,        # accumulator threshold — lower = more circles detected
        minRadius=20,     # smallest coin radius to detect
        maxRadius=80      # largest coin radius to detect
    )
    if circles is None:
        return []
    return np.round(circles[0, :]).astype("int")  # list of (x, y, radius)

def check_overlap(circles):
    # returns True if any two circles intersect
    for i in range(len(circles)):
        for j in range(i+1, len(circles)):
            x1, y1, r1 = circles[i]
            x2, y2, r2 = circles[j]
            dist = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            if dist < r1 + r2:
                return True
    return False
```

---

### Step 4 — Crop Each Coin ROI (`utils.py`)

After detection, extract each coin as an individual image patch.

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

Runs **per crop** after detection. Cleans the coin mask and handles overlaps.

```python
import cv2
import numpy as np

def clean_coin_mask(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    # closing: fills small holes inside coin
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    # opening: removes small noise outside coin
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=1)
    return opened

def watershed_separation(image, circles):
    # used when check_overlap() returns True
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    sure_bg = cv2.dilate(binary, kernel, iterations=3)
    dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)
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

Runs on each cleaned coin crop. Produces a feature vector for classification.

```python
import cv2
import numpy as np

def extract_color_features(crop):
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    # mean HSV values
    mean_h, mean_s, mean_v = cv2.mean(hsv)[:3]
    # HSV histogram (8 bins per channel)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8],
                        [0, 180, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return mean_h, mean_s, mean_v, hist

def extract_texture_features(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    # LBP-like: use Laplacian variance as texture measure
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var

def extract_features(crop, radius):
    mean_h, mean_s, mean_v, hist = extract_color_features(crop)
    texture = extract_texture_features(crop)
    feature_vector = np.concatenate([
        [radius],                        # 1 value  — size
        [mean_h, mean_s, mean_v],        # 3 values — mean color
        hist,                            # 512 values — color distribution
        [texture]                        # 1 value  — texture
    ])
    return feature_vector
```

---

### Step 7 — Classification (`classify.py`)

Three independent votes combined into a majority decision.

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
    cv2.putText(frame, f"Total: {total:.2f}",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 200, 255), 2)
    return frame
```

---

## 6. Phase 2 — SVM Upgrade

Only do this if Phase 1 voting accuracy is below ~90% on your test images.

### Why SVM helps
Instead of three separate votes that can disagree, SVM learns the *combination* of all features together during training. It finds the boundary in feature space that best separates each coin type — far more robust than voting under varied lighting or with mixed currencies.

### Data collection
- Photograph each coin denomination ~100 times
- Vary lighting, angle, and background
- Label each image with its denomination string
- Target: 100 images × number of coin types

### Training (`07_svm_training.ipynb`)

```python
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import numpy as np

# X = feature vectors, y = labels
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

clf = SVC(kernel='rbf', C=10, gamma='scale', probability=True)
clf.fit(X_train, y_train)

print(classification_report(y_test, clf.predict(X_test)))
joblib.dump((clf, scaler), 'models/svm_coin_classifier.pkl')
```

### Drop-in replacement (`svm_classifier.py`)

Same function signature as `classify.py` — swap with one import change:

```python
import joblib
import numpy as np
from features import extract_features

def classify_coin(crop, radius, templates=None):
    clf, scaler = joblib.load('models/svm_coin_classifier.pkl')
    features = extract_features(crop, radius)
    features_scaled = scaler.transform([features])
    label = clf.predict(features_scaled)[0]
    confidence = clf.predict_proba(features_scaled).max()
    return label, confidence
```

### One-line swap in `pipeline.py` and `realtime.py`

```python
# from classify import classify_coin       # Phase 1 voting
from svm_classifier import classify_coin   # Phase 2 SVM
```

---

## 7. Phase 3 — Real-Time Webcam

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
import numpy as np
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
templates = {}  # load templates dict here

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # step 2: preprocess
    preprocessed = preprocess(frame)

    # step 3: detect — run on downscaled for speed
    small = resize_for_detection(preprocessed, scale=0.5)
    circles_small = detect_coins(small)
    # scale coordinates back to original resolution
    circles = [(x*2, y*2, r*2) for (x, y, r) in circles_small]

    # step 4-7: per coin
    labels, confidences = [], []
    for (x, y, r) in circles:
        crop = crop_roi(frame, x, y, r)
        if crop.size == 0:
            continue
        # step 5: morphology
        clean_coin_mask(crop)
        # step 6: features
        features = extract_features(crop, r)
        # step 7: classify
        label, conf = classify_coin(crop, r, templates)
        labels.append(label)
        confidences.append(conf)

    # stability check
    stability_buffer.append(len(labels))
    if len(set(stability_buffer)) == 1:
        locked_total = sum(COIN_VALUES.get(l, 0) for l in labels)
        locked_labels = labels

    # step 9: draw overlay
    frame = draw_overlay(frame, circles, locked_labels, confidences, locked_total)
    cv2.imshow("Coin Counter", frame)

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

## 8. Notebooks Breakdown

### `01_data_collection.ipynb`
- Photograph each coin denomination under 3+ lighting conditions
- Capture 100+ images per coin type for SVM dataset
- Build `data/templates/` — one clean photo per coin × 12 rotations (every 30°)
- Run `calibration.py` interactively and verify saved values

### `02_preprocessing.ipynb`
- Experiment with blur kernel sizes (3×3, 5×5, 9×9)
- Compare plain grayscale vs CLAHE for reflective coins
- Visualize effect of each step on sample images
- Save best parameters as constants

### `03_detection.ipynb`
- Tune all `HoughCircles` parameters interactively with sliders
- Test on images with 1, 3, 5, and 10 coins
- Test overlapping coins — visualize pre/post watershed
- Document final parameter values for `detect.py`

### `04_morphology.ipynb`
- Apply closing, opening, dilation, erosion to coin crops
- Visualize which operations clean coin masks best
- Test watershed on intentionally overlapping coin images
- Document which kernel size and iteration count works best

### `05_feature_extraction.ipynb`
- Extract and visualize HSV histograms per coin type
- Plot radius distribution per denomination
- Show texture (Laplacian variance) differences between coin types
- Confirm features are separable before classification

### `06_classification.ipynb`
- Test size vote alone: accuracy per denomination
- Test color vote alone: accuracy per denomination
- Test template matching alone: accuracy, effect of rotation
- Combine: majority vote accuracy
- Document which coin pairs get confused

### `07_svm_training.ipynb`
- Load labeled image dataset
- Extract feature vectors for all images
- Split train/test 80/20
- Train SVM, print `classification_report`
- Compare accuracy to Phase 1 voting
- Save model to `models/svm_coin_classifier.pkl`

### `08_evaluation.ipynb`
- Side-by-side: voting accuracy vs SVM accuracy per class
- Confusion matrix heatmap for both approaches
- Per-denomination precision, recall, F1
- Failure analysis: which coins are still confused and why
- Final summary table for your written report

---

## 9. Scripts Breakdown

### `calibration.py`
Run once before everything else. Opens camera, user places one coin at a time, presses spacebar to capture. Script detects coin, measures radius and HSV, saves to `coin_map.py` automatically.

### `preprocess.py`
- `preprocess(image)` → grayscale + blur + CLAHE → returns cleaned image
- `resize_for_detection(image, scale)` → downscale for real-time speed

### `detect.py`
- `detect_coins(preprocessed)` → returns list of `(x, y, r)`
- `check_overlap(circles)` → returns True if any two circles intersect

### `morphology.py`
- `clean_coin_mask(crop)` → closing + opening on binary coin mask
- `watershed_separation(image, circles)` → separates overlapping coins

### `features.py`
- `extract_color_features(crop)` → mean HSV + histogram
- `extract_texture_features(crop)` → Laplacian variance
- `extract_features(crop, radius)` → full feature vector (for SVM)

### `classify.py`
- `size_vote(radius)` → label string
- `color_vote(crop)` → label string
- `template_vote(crop, templates)` → label string
- `classify_coin(crop, radius, templates)` → `(label, confidence)`

### `svm_classifier.py`
- `classify_coin(crop, radius, templates=None)` → `(label, confidence)`
- Same interface as `classify.py` — one import swap to switch

### `coin_map.py`
Three dicts: `COIN_SIZES`, `COIN_COLORS`, `COIN_VALUES`.
Add new currencies by extending these dicts only — no other file changes needed.

### `pipeline.py`
Accepts an image path, runs all 9 steps, saves annotated output to `outputs/`.

### `realtime.py`
Full webcam loop. Press `q` to quit, `s` to save snapshot.
Stability buffer of 5 frames before locking and displaying total.

### `utils.py`
- `crop_roi(image, x, y, r)` → square crop around detected circle
- `draw_overlay(frame, circles, labels, confidences, total)` → draws all annotations
- `extract_hsv_mean(crop)` → returns mean H, S, V values

---

## 10. Calibration Guide

The most critical step. Wrong calibration = wrong everything downstream.

**Requirements:**
- Fixed camera position — do not move it after calibration
- Flat, matte surface (white or dark paper — avoid gloss)
- Even, diffuse lighting — avoid direct sunlight or single-point lamps
- One coin at a time, flat on the surface

**Steps:**
1. Set up your camera at its final shooting position
2. Run `python calibration.py`
3. Place each denomination in frame, press spacebar to capture
4. Repeat for all coin types across all currencies
5. Verify saved values in `coin_map.py` look reasonable
6. Run `pipeline.py` on 3 test photos to verify

**If results are wrong later:** Re-run calibration under the same lighting as your demo.

---

## 11. Known Challenges & Fixes

| Challenge | Why it happens | Fix |
|---|---|---|
| Coins not detected | `param2` too high | Lower `param2` to 25–30 |
| False circles detected | Background texture | Raise `param2`, use plain background |
| Overlapping coins merged | HoughCircles treats two as one | Watershed fallback in `morphology.py` |
| 0.50 and 0.25 EGP confused | Similar size and color | Add SVM (Phase 2) |
| Template matching fails on rotation | Coin is rotated | Store 12 rotations per template |
| Total flickers in real-time | Frame-to-frame variation | Stability buffer (N=5 frames) |
| Slow real-time performance | Full-res HoughCircles is slow | Downscale to 50% before detection |
| Glare on metallic coins | Reflective surface | Use diffuse/indirect lighting |
| Camera moved between sessions | Pixel radius changes | Recalibrate at new position |
| Unknown coin label | All three votes disagree | Draw `?`, log to CSV for review |

---

## 12. Evaluation Metrics

### Phase 1 — voting
- Per-denomination classification accuracy
- Overall accuracy across all coin types
- Confusion matrix (which pairs get confused most)

### Phase 2 — SVM
- Same metrics directly compared to Phase 1
- Precision, recall, F1 per class (`classification_report`)
- Training vs test accuracy (check for overfitting)

### Phase 3 — real-time
- Average FPS (`cv2.getTickCount()` measurement)
- Frames until total locks in (stability latency)
- Qualitative: performance under different lighting

### For your written report
Present a table comparing voting vs SVM accuracy per denomination.
Show confusion matrix for both. Explain where each approach fails and why.
Include sample annotated output images.

---

## 13. Suggested Timeline

| Week | Work |
|---|---|
| Week 1 | Set up project structure, run calibration, build templates folder |
| Week 1 | Complete notebooks 01 (data collection) and 02 (preprocessing) |
| Week 2 | Complete notebooks 03 (detection) and 04 (morphology) |
| Week 2 | Complete notebook 05 (feature extraction) — verify features are separable |
| Week 3 | Complete notebook 06 (classification) — voting pipeline end-to-end |
| Week 3 | Collect labeled dataset, complete notebook 07 (SVM training) |
| Week 4 | Complete notebook 08 (evaluation), compare voting vs SVM |
| Week 4 | Build `realtime.py`, tune stability buffer, prepare demo |

> **If time is short:** Stop after Week 2–3. A working static pipeline with voting is a complete,
> submittable project. Phase 2 (SVM) and Phase 3 (real-time) are upgrades, not requirements.

---

*Generated for CSE483: Computer Vision — Ain Shams University*
