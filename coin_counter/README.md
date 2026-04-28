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
