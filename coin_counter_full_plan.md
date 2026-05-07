# Coin Counting System — Updated Project Plan
**Course:** CSE483: Computer Vision — Ain Shams University  
**Implementation status:** Live webcam counter for Egyptian coins using classical computer vision  
**Currency:** Egyptian Pounds (EGP)

---

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [Actual Pipeline](#2-actual-pipeline)
3. [Supported Denominations](#3-supported-denominations)
4. [Actual Folder Structure](#4-actual-folder-structure)
5. [Libraries & Requirements](#5-libraries--requirements)
6. [Current Implementation](#6-current-implementation)
7. [Support Modules](#7-support-modules)
8. [Future Work](#8-future-work)
9. [Known Challenges & Fixes](#9-known-challenges--fixes)
10. [Evaluation Metrics](#10-evaluation-metrics)
11. [Recommended Next Steps](#11-recommended-next-steps)

---

## 1. Project Overview

This repository currently implements a live webcam-based Egyptian coin counter.
The active runtime is in `coin_counter/src/main.py`, which detects coin circles and classifies them using classical CV heuristics.

The working goal today is:
- detect coins using Hough circles,
- filter false / nested circles,
- distinguish between `1 EGP` and `50 PT`,
- display the total value on a live stacked view.

---

## 2. Actual Pipeline

The current implementation follows this sequence in `main.py`:

1. Capture webcam frame from `cv.VideoCapture(0)`
2. Convert BGR frame to grayscale
3. Apply Gaussian blur
4. Apply Otsu thresholding to obtain a binary mask
5. Apply morphological closing to clean the mask
6. Convert the original frame to HSV and threshold the brass/silver color range
7. Combine the binary mask and HSV mask with `bitwise_and`
8. Run `cv.HoughCircles` on the combined mask
9. Filter inner / nested circles with `filter_inner_circles()`
10. Classify each detected coin by radius threshold
11. Draw annotations and total using `cvzone`

> This is the actual running pipeline, not the larger planned pipeline from earlier scaffolding.

---

## 3. Supported Denominations

The active code currently supports two coin denominations:

| Label | Value | Current support |
|---|---|---|
| `EGP_1` | 1.00 EGP | Supported |
| `EGP_0.50` | 0.50 EGP | Supported |

### Not supported yet

- `EGP_0.25` (25 piastres) is not implemented in the current runtime.

### Current classification rule

In `coin_counter/src/main.py`:

```python
RADIUS_THRESHOLD = 55

def classify_coin(radius):
    if radius < RADIUS_THRESHOLD:
        return 1
    else:
        return 0.5
```

This means the current classifier is a single size-based rule, not a learned model.

---

## 4. Actual Folder Structure

```
coin_counter/
├── requirements.txt
├── README.md
├── src/
│   ├── __init__.py
│   ├── coin_map.py
│   ├── detect.py
│   ├── features.py
│   ├── main.py
│   ├── morphology.py
│   ├── preprocess.py
│   ├── project.py
│   ├── utils.py
│   └── outputs/

EGP Coin Dataset/
├── batch1/
├── batch2/
├── batch3/
└── batch4/

setup_coin_counter.sh
coin_counter_full_plan.md
README.md
```

> `coin_counter/src/project.py` contains commented-out design notes and is not the active runtime.

---

## 5. Libraries & Requirements

### Actual dependencies used by `main.py`

- `opencv-python`
- `numpy`
- `cvzone`

### `coin_counter/requirements.txt`

The current requirements file contains a longer planned dependency list, but the active runtime only needs the three packages above.

```text
opencv-python>=4.8.0
numpy>=1.24.0
scikit-learn>=1.3.0
joblib>=1.3.0
matplotlib>=3.7.0
pandas>=2.0.0
jupyter>=1.0.0
```

### Important note

`cvzone` is required by the current live demo but is not listed in `requirements.txt`.
Add it before running `coin_counter/src/main.py`.

---

## 6. Current Implementation

### `coin_counter/src/main.py`

This is the working application.
It reads webcam frames and runs the detection/classification pipeline live.

Key components in `main.py`:
- live parameter tuning with OpenCV trackbars for `HoughP2`, `MinRadius`, `MaxRadius`
- HSV filtering for coin color range
- binary mask cleaning with morphology
- Hough circle detection
- nested-circle filtering to discard inner edge circles
- annotation using `cvzone.stackImages`

### `coin_counter/src/coin_map.py`

Actual content:

```python
COIN_SIZES = {
    "EGP_1":    52,
    "EGP_0.50": 45,
}

COIN_COLORS = {
    "EGP_1":    {"h": 20, "s": 120, "v": 170},
    "EGP_0.50": {"h": 22, "s": 160, "v": 155},
}

COIN_VALUES = {
    "EGP_1":    1.00,
    "EGP_0.50": 0.50,
}
```

`coin_map.py` currently contains only two denominations and is not actively used by `main.py`.

---

## 7. Support Modules

The repo contains reusable modules that are not integrated into `main.py` yet.
These are useful for future refactoring and a full static pipeline.

- `preprocess.py` — grayscale, blur, CLAHE, crop preprocessing
- `detect.py` — HoughCircles wrapper, filtering, overlap detection
- `morphology.py` — mask cleaning, watershed separation, region extraction
- `features.py` — color, texture, and shape feature extraction
- `utils.py` — ROI crop, HSV mean, overlay drawing, template loading

These modules are good starting points for converting the current live demo into a modular pipeline.

---

## 8. Future Work

The current repo includes scaffolding for a more advanced design, but the active code is simpler.
Use this as a roadmap for next development:

- Add 25 piastres support (`EGP_0.25`)
- Wire `detect.py`, `morphology.py`, `features.py`, and `utils.py` into an actual `pipeline.py`
- Replace the hard threshold classifier with a multi-feature voting or SVM classifier
- Add template matching support if desired
- Use the `EGP Coin Dataset/` folders for training and evaluation
- Update `requirements.txt` to include `cvzone` and remove unused extras if not used

---

## 9. Known Challenges & Fixes

| Challenge | Why it happens | Fix |
|---|---|---|
| Live Hough detection misses coins | weak mask or poor lighting | tune HSV range and Hough params |
| False circles from noise | background texture or internal features | strengthen morphology / raise `param2` |
| Nested inner circles | Hough sees ring/edge regions | use `filter_inner_circles()` |
| 1 EGP vs 50 PT confusion | hard threshold too strict for some shots | adjust radius threshold or add color/texture features |
| Missing 25 PT support | repository currently only has two denominations | implement 25 PT branch in the classifier |
| Missing `cvzone` dependency | runtime uses it but `requirements.txt` omits it | add `cvzone` to requirements |

---

## 10. Evaluation Metrics

For the current active code, evaluate:

- detection rate: how many coins are found in live frames
- false positive rate: detected objects that are not coins
- classification accuracy between `1 EGP` and `50 PT`
- stability of the live overlay
- sensitivity to lighting and background changes

If you extend the repo later, add:
- per-class precision / recall
- confusion matrix for all coin classes
- runtime FPS and latency

---

## 11. Recommended Next Steps

1. Install runtime dependencies:
```bash
pip install opencv-python numpy cvzone
```
2. Run the live demo:
```bash
python coin_counter/src/main.py
```
3. Tune the live trackbars for your camera and coins.
4. Add `cvzone` to `coin_counter/requirements.txt`.
5. Refactor the current code to use the support modules and implement `EGP_0.25`.
