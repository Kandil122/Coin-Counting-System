#!/bin/bash

# ============================================================
#  Coin Counter — Project Setup Script
#  CSE483: Computer Vision — Ain Shams University
#  Run: bash setup_coin_counter.sh
# ============================================================

echo "Creating coin_counter project structure..."

# ── Root folder ──
mkdir -p coin_counter

# ── Data folders ──
mkdir -p coin_counter/data/raw
mkdir -p coin_counter/data/processed
mkdir -p coin_counter/data/templates

# ── Notebooks ──
mkdir -p coin_counter/notebooks
touch coin_counter/notebooks/01_data_collection.ipynb
touch coin_counter/notebooks/02_preprocessing.ipynb
touch coin_counter/notebooks/03_detection.ipynb
touch coin_counter/notebooks/04_morphology.ipynb
touch coin_counter/notebooks/05_feature_extraction.ipynb
touch coin_counter/notebooks/06_classification.ipynb
touch coin_counter/notebooks/07_svm_training.ipynb
touch coin_counter/notebooks/08_evaluation.ipynb

# ── Source scripts ──
mkdir -p coin_counter/src
touch coin_counter/src/__init__.py
touch coin_counter/src/preprocess.py
touch coin_counter/src/detect.py
touch coin_counter/src/morphology.py
touch coin_counter/src/features.py
touch coin_counter/src/classify.py
touch coin_counter/src/svm_classifier.py
touch coin_counter/src/coin_map.py
touch coin_counter/src/pipeline.py
touch coin_counter/src/realtime.py
touch coin_counter/src/utils.py

# ── Models folder ──
mkdir -p coin_counter/models

# ── Outputs folder ──
mkdir -p coin_counter/outputs

# ── Root files ──
touch coin_counter/calibration.py
touch coin_counter/README.md

# ── requirements.txt ──
cat > coin_counter/requirements.txt << 'EOF'
opencv-python>=4.8.0
numpy>=1.24.0
scikit-learn>=1.3.0
joblib>=1.3.0
matplotlib>=3.7.0
pandas>=2.0.0
jupyter>=1.0.0
EOF

# ── coin_map.py starter content ──
cat > coin_counter/src/coin_map.py << 'EOF'
# ============================================================
#  coin_map.py
#  Populated by calibration.py — edit values after calibrating
# ============================================================

COIN_SIZES = {
    "EGP_1":       52,
    "EGP_0.50":    45,
    "EGP_0.25":    38,
    "USD_quarter":  47,
    "USD_dime":     38,
    "EUR_1":        54,
}

COIN_COLORS = {
    "EGP_1":       {"h": 25, "s": 180, "v": 160},
    "EGP_0.50":    {"h": 0,  "s": 20,  "v": 200},
    "EGP_0.25":    {"h": 15, "s": 60,  "v": 180},
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
EOF

# ── README.md starter content ──
cat > coin_counter/README.md << 'EOF'
# Coin Counter — CSE483 Computer Vision

## Setup
```bash
pip install -r requirements.txt
```

## Steps
1. Run calibration first:
```bash
python calibration.py
```

2. Run static pipeline on an image:
```bash
python src/pipeline.py --image data/raw/test.jpg
```

3. Run real-time webcam mode:
```bash
python src/realtime.py
```

## Controls (real-time mode)
- `q` — quit
- `s` — save snapshot to outputs/

## Project Structure
- `data/raw/`        — original coin photos
- `data/templates/`  — reference coin images (12 rotations each)
- `notebooks/`       — one notebook per pipeline stage
- `src/`             — clean Python scripts
- `models/`          — saved SVM model after training
- `outputs/`         — annotated result images
EOF

# ── Print result ──
echo ""
echo "Done! Project structure created:"
echo ""
find coin_counter -type f | sort
echo ""
echo "Next steps:"
echo "  1. cd coin_counter"
echo "  2. pip install -r requirements.txt"
echo "  3. python calibration.py"
