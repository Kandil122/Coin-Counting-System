import cv2
import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# PARAMETERS (Measured Averages / Reference Values)
# =========================================================
coin_params = {
    "EGP_1": {"radius": 80, "h": 70, "s": 40, "v": 160},
    "EGP_0.50": {"radius": 77, "h": 41, "s": 60, "v": 191}
}

# =========================================================
# AUTO MODE DETECTION THRESHOLD
# =========================================================
DARK_V_THRESHOLD = 75  # median V below this => very_dark mode

# =========================================================
# HOUGH PARAMETERS
# =========================================================
minDist = 60
param2 = 30

# =========================================================
# HELPERS
# =========================================================
def filter_inner_circles(circles, center_thresh=35):
    if circles is None:
        return []

    circles = np.uint16(np.around(circles[0]))
    circles = sorted(circles, key=lambda c: c[2], reverse=True)

    filtered = []
    for x, y, r in circles:
        keep = True
        for fx, fy, fr in filtered:
            dist = np.sqrt((x - fx) ** 2 + (y - fy) ** 2)
            if dist < center_thresh:
                keep = False
                break
        if keep:
            filtered.append((x, y, r))

    return filtered


def safe_roi(hsv, x, y, r):
    h, w = hsv.shape[:2]
    x1 = max(0, x - r)
    x2 = min(w, x + r)
    y1 = max(0, y - r)
    y2 = min(h, y + r)

    roi = hsv[y1:y2, x1:x2]
    if roi.size == 0:
        return None

    return roi


def classify_coin(r, mean_h, mean_s, mean_v):
    # Strong radius decision
    if r <= 73:
        return 1.0
    if r >= 82:
        return 1.0

    # Ambiguous zone => use HSV
    if mean_v > 185 and mean_s > 50:
        return 0.5

    return 1.0


# =========================================================
# IMAGE LOAD
# =========================================================
imagePath = "/home/adham/Documents/Coin-Counting-System/EGP Coin Dataset/batch2/coin10.jpeg"
image = cv2.imread(imagePath)

if image is None:
    print("Image failed to load. Check the path:", imagePath)
    exit()

imageCopy = image.copy()

# =========================================================
# MODE DETECTION (NORMAL vs VERY_DARK)
# =========================================================
hsv_full = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
median_v = np.median(hsv_full[:, :, 2])

if median_v < DARK_V_THRESHOLD:
    mode = "very_dark"
else:
    mode = "normal"

print("Detected Mode:", mode)
print("Median V:", median_v)

# =========================================================
# PIPELINE
# =========================================================
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# --------------------------------
# VERY DARK MODE PREPROCESSING
# --------------------------------
if mode == "very_dark":
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    blur = cv2.GaussianBlur(gray, (7, 7), 0)

    edges = cv2.Canny(blur, 20, 100)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

# --------------------------------
# NORMAL MODE PREPROCESSING
# --------------------------------
else:
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blur, 40, 120)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

# =========================================================
# HOUGH CIRCLE DETECTION
# =========================================================
circles = cv2.HoughCircles(
    blur,
    cv2.HOUGH_GRADIENT,
    dp=1.2,
    minDist=minDist,
    param1=120,
    param2=param2,
    minRadius=71,
    maxRadius=80
)

filtered_circles = filter_inner_circles(circles, center_thresh=35)

# =========================================================
# CLASSIFICATION + DRAW
# =========================================================
coins = []
total = 0

for x, y, r in filtered_circles:
    roi = safe_roi(hsv_full, x, y, r)
    if roi is None:
        continue

    mean_h = np.mean(roi[:, :, 0])
    mean_s = np.mean(roi[:, :, 1])
    mean_v = np.mean(roi[:, :, 2])

    # if very_dark => ignore hsv classification
    if mode == "very_dark":
        value = 1.0 if r >= 74 else 0.5
    else:
        value = classify_coin(r, mean_h, mean_s, mean_v)

    total += value
    coins.append((x, y, r, value))

    cv2.circle(imageCopy, (x, y), r, (0, 255, 0), 2)
    cv2.circle(imageCopy, (x, y), 2, (0, 0, 255), 3)

    cv2.putText(
        imageCopy,
        f"{value} EGP",
        (x - 40, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 0, 0),
        2
    )

    print(f"R={r} | H={mean_h:.1f} S={mean_s:.1f} V={mean_v:.1f} => {value} EGP")

print("Detected coins:", len(coins))
print("Total money:", total)

# =========================================================
# DISPLAY RESULTS
# =========================================================
plt.figure(figsize=(16, 10))

plt.subplot(231)
plt.imshow(image[:, :, ::-1])
plt.title("Original")
plt.axis("off")

plt.subplot(232)
plt.imshow(gray, cmap="gray")
plt.title("Gray / CLAHE if Dark")
plt.axis("off")

plt.subplot(233)
plt.imshow(edges, cmap="gray")
plt.title("Edges")
plt.axis("off")

plt.subplot(234)
temp_hough = image.copy()
if circles is not None:
    for c in circles[0]:
        cx, cy, cr = c
        cv2.circle(temp_hough, (int(cx), int(cy)), int(cr), (0, 255, 0), 2)
plt.imshow(temp_hough[:, :, ::-1])
plt.title("Raw Hough Circles")
plt.axis("off")

plt.subplot(235)
plt.imshow(imageCopy[:, :, ::-1])
plt.title(f"Final Detection | Total = {total}")
plt.axis("off")

plt.tight_layout()
plt.show()
