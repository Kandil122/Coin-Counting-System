# ============================================================
#  coin_map.py
#  EGP denomination reference — populated by calibration.py
#  Edit COIN_SIZES and COIN_COLORS after running calibration
# ============================================================

# Radius in pixels at YOUR fixed camera height
# Run calibration.py to measure these for your setup
COIN_SIZES = {
    "EGP_1":    52,    # 1 Pound   — 23mm
    "EGP_0.50": 45,    # 50 Piastres — 21mm
}

COIN_COLORS = {
    "EGP_1":    {"h": 20, "s": 120, "v": 170},  # bimetallic — silver ring lowers saturation
    "EGP_0.50": {"h": 22, "s": 160, "v": 155},  # fully brass — high saturation throughout
}

COIN_VALUES = {
    "EGP_1":    1.00,
    "EGP_0.50": 0.50,
}

COIN_LABELS = {
    "EGP_1":    "1 EGP",
    "EGP_0.50": "50 PT",
    "unknown":  "???",
}
# All valid denomination keys
ALL_COINS = list(COIN_VALUES.keys())