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
