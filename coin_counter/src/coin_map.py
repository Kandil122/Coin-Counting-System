# ============================================================
#  coin_map.py
#  EGP denomination reference — populated by calibration.py
#  Edit COIN_SIZES and COIN_COLORS after running calibration
# ============================================================

# Radius in pixels at YOUR fixed camera height
# Run calibration.py to measure these for your setup
COIN_SIZES = {
    "EGP_1":    52,    # 1 Pound   — largest  (23mm real diameter)
    "EGP_0.50": 45,    # 50 Piastres — medium (21mm real diameter)
    "EGP_0.25": 38,    # 25 Piastres — smallest(18mm real diameter)
}

# Mean HSV values per denomination at your lighting setup
# Run calibration.py to measure these for your setup
COIN_COLORS = {
    "EGP_1":    {"h": 25, "s": 180, "v": 160},   # gold / brass tone
    "EGP_0.50": {"h": 0,  "s": 20,  "v": 200},   # silver tone
    "EGP_0.25": {"h": 15, "s": 60,  "v": 180},   # silver with copper edge
}

# Face value in EGP
COIN_VALUES = {
    "EGP_1":    1.00,
    "EGP_0.50": 0.50,
    "EGP_0.25": 0.25,
}

# Human-readable display labels
COIN_LABELS = {
    "EGP_1":    "1 EGP",
    "EGP_0.50": "50 PT",
    "EGP_0.25": "25 PT",
    "unknown":  "???",
}

# All valid denomination keys
ALL_COINS = list(COIN_VALUES.keys())