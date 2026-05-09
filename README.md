# Egyptian Coin Detection & Classification System
### CSE386 Computer Vision Project

A **computer vision pipeline** for automated detection, classification, and monetary value calculation of Egyptian coins (EGP 1 and EGP 0.50) using OpenCV and Hough Circle Transform.

---

## 📌 Quick Start

### Prerequisites
```
opencv-python>=4.8.0
numpy>=1.24.0
scikit-learn>=1.3.0
joblib>=1.3.0
matplotlib>=3.7.0
pandas>=2.0.0
jupyter>=1.0.0
```

### Run Full Pipeline (Image Analysis)
```bash
python full_pipeline.py
```
Analyzes a single image, detects coins, classifies them, and outputs the total monetary value with visualization.

### Run Live Detection (Camera Test)
```bash
python live_detection.py
```
Activates your camera for **real-time coin detection**. Press `q` to exit.

---

## 📁 Project Structure

```
Coin-Counting-System/
│
├── full_pipeline.py           # Main pipeline - loads image, detects & classifies coins
├── live_detection.py          # Real-time camera-based detection
│
└── EGP Coin Dataset/
    ├── batch1/                # Training/testing images (Batch 1)
    ├── batch2/                # Training/testing images (Batch 2)
    ├── batch3/                # Training/testing images (Batch 3)
    └── batch4/                # Training/testing images (Batch 4)
```

---

## 📊 Datasets

The **EGP Coin Dataset** contains multiple batches of real-world images with:
- **EGP 1 coins** (radius ~80 px)
- **EGP 0.50 coins** (radius ~77 px)
- **Varying lighting conditions** (normal, dark, outdoor)
- **Multiple coin arrangements** (single coins, groups, overlaps)

### Dataset Statistics
- **Total images**: 100+ real photos
- **Denominations**: 2 types (EGP 1, EGP 0.50)
- **Image quality**: 720p–4K resolution
- **Lighting scenarios**: Indoor normal, very dark, natural light

Each batch can be used for:
- Training and validation
- Testing detection accuracy
- Benchmarking classification performance

---

## 🏗️ Architecture

### Pipeline Overview

```
Input Image
    ↓
[STAGE 1] Lighting Detection
    ├─ Calculate median brightness (V channel in HSV)
    └─ Choose preprocessing mode: "normal" or "very_dark"
    ↓
[STAGE 2] Preprocessing
    ├─ Convert to Grayscale
    ├─ If very_dark: Apply CLAHE (contrast enhancement)
    ├─ Gaussian Blur (remove noise)
    └─ Canny Edge Detection → Dilation
    ↓
[STAGE 3] Circle Detection (Hough Transform)
    ├─ Detect circular shapes (radius: 71–80 px)
    ├─ Return candidate circles
    └─ Each circle: (center_x, center_y, radius)
    ↓
[STAGE 4] Filter Overlaps
    ├─ Sort circles by radius (largest first)
    ├─ Remove circles closer than 35 px to neighbors
    └─ Return cleaned detection list
    ↓
[STAGE 5] Classification (HSV-based)
    ├─ Extract ROI (Region of Interest) for each coin
    ├─ Calculate mean Hue, Saturation, Value
    ├─ If very_dark mode: Use radius only
    ├─ Else: Use radius + HSV statistics
    └─ Output: Coin value (1.0 EGP or 0.5 EGP)
    ↓
[STAGE 6] Aggregation & Output
    ├─ Sum all coin values
    ├─ Draw detections on image
    └─ Visualize 6-panel diagnostic view
    ↓
Output: Total Money, Coin Count, Visual Overlay
```


**Coin Reference Data:**
| Coin | Radius | Hue | Saturation | Value |
|------|--------|-----|-----------|-------|
| EGP 1 | 80 | 70 | 40 | 160 |
| EGP 0.50 | 77 | 41 | 60 | 191 |

**Why HSV instead of RGB?**
- **Hue (H)**: Color tone, independent of lighting
- **Saturation (S)**: Color intensity (distinguishes coin types)
- **Value (V)**: Brightness (adjusts for lighting variations)

#### **6. Aggregation**
- Sum all coin values: `total = Σ(coin_value)`
- Generate 6-panel visualization:
  1. Original image
  2. Preprocessed (grayscale / CLAHE)
  3. Edge map
  4. Raw Hough detections
  5. Final filtered & classified coins
  6. Diagnostic output

---

## 🔄 Data Flow Diagram

```
Image Input
    │
    ├─ HSV Conversion
    │  └─ Brightness Analysis (Mode Detection)
    │
    ├─ Grayscale Conversion
    │  └─ Adaptive Preprocessing
    │
    ├─ Edge Detection
    │  └─ Morphological Operations
    │
    ├─ Hough Circle Detection
    │  └─ Returns: [center_x, center_y, radius]
    │
    ├─ Overlap Filtering
    │  └─ Removes Duplicates
    │
    ├─ ROI Extraction (per coin)
    │  └─ HSV Statistics Calculation
    │
    ├─ Classification Engine
    │  ├─ Radius-based decision
    │  └─ HSV-based tiebreaker
    │
    ├─ Aggregation
    │  └─ Sum Values, Generate Output
    │
    └─ Visualization & Display
       └─ 6-panel diagnostic grid
```

---

## ⚙️ Key Parameters

### Tunable Settings (in code)

| Parameter | Default | Effect | When to Change |
|-----------|---------|--------|-----------------|
| `DARK_V_THRESHOLD` | 75 | Brightness threshold for mode detection | Very dark/bright images |
| `minDist` | 60 | Min distance between circle centers | Coins closer/farther apart |
| `param2` | 30 | Hough accumulator threshold | More/fewer detections |
| `minRadius` | 71 | Min coin size (px) | Coins appear smaller |
| `maxRadius` | 80 | Max coin size (px) | Coins appear larger |
| `center_thresh` | 35 | Min distance to keep circles | Overlapping coins |

---

## 🎯 Strengths & Limitations

### ✅ Strengths
- **Adaptive**: Automatically detects lighting and adjusts preprocessing
- **Fast**: Real-time capable (Hough is O(n) in image size)
- **Robust**: Works with partial occlusion, varied lighting, camera angles
- **Simple**: No ML training required, purely algorithmic
- **Interpretable**: Each step can be visualized and debugged

### ⚠️ Limitations
- **Radius assumption**: Coins must be roughly consistent size (no extreme perspective)
- **Color-dependent**: Requires good lighting for HSV classification
- **Clutter sensitivity**: Works best with isolated coins or sparse scenes
- **Single denomination**: Designed for 2 Egyptian denominations only
- **No confidence scoring**: Binary classification, not probabilistic

---

## 🚀 Future Enhancements

- [ ] **CNN Classifier**: Train deep learning model for denomination detection
- [ ] **Confidence Scores**: Output probability per detection
- [ ] **Multi-Currency**: Support USD, EUR, GBP coins
- [ ] **Perspective Correction**: Handle tilted/angled coins
- [ ] **Overlapping Coins**: Segment and separate touching coins
- [ ] **Mobile Deployment**: TensorFlow Lite for smartphones
- [ ] **Web Interface**: Flask/Streamlit app for easy testing

---

## 📋 Example Output

```
Detected Mode: normal
Median V: 140

R=80 | H=71.2 S=38.5 V=162.0 => 1.0 EGP
R=77 | H=40.8 S=62.1 V=193.5 => 0.5 EGP
R=79 | H=69.5 S=39.2 V=161.0 => 1.0 EGP

Detected coins: 3
Total money: 2.5 EGP
```

---



---

## 🤝 Contributing

Contributions welcome! Please submit issues or improvements.
