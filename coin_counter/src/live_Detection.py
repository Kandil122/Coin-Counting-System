import cv2 as cv
import cvzone
import numpy as np

# ================================
# CAMERA SETUP
# ================================
cap = cv.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# ================================
# HSV RANGE (the one we used before)
# ================================
lowerHSV = np.array([0, 0, 120])
upperHSV = np.array([179, 80, 255])

# ================================
# TRACKBARS
# ================================
def empty(a):
    pass

cv.namedWindow("Settings")
cv.resizeWindow("Settings", 640, 300)

cv.createTrackbar("Canny1", "Settings", 27, 255, empty)
cv.createTrackbar("Canny2", "Settings", 120, 255, empty)

# Hough parameters (so you can tune live)
cv.createTrackbar("HoughP2", "Settings", 35, 100, empty)     # param2
cv.createTrackbar("MinRadius", "Settings", 45, 200, empty)
cv.createTrackbar("MaxRadius", "Settings", 90, 300, empty)

# ================================
# FILTER INNER CIRCLES (IMPORTANT)
# ================================
def filter_inner_circles(circles, center_thresh=30):
    if circles is None:
        return []

    circles = np.uint16(np.around(circles[0]))
    circles = sorted(circles, key=lambda c: c[2], reverse=True)  # sort by radius desc

    filtered = []
    for x, y, r in circles:
        keep = True
        for fx, fy, fr in filtered:
            dist = np.sqrt((x - fx) ** 2 + (y - fy) ** 2)
            if dist < center_thresh:  # same center => inner circle
                keep = False
                break
        if keep:
            filtered.append((x, y, r))

    return filtered

# ================================
# CLASSIFY COINS (0.5 and 1 EGP)
# ================================
RADIUS_THRESHOLD = 55

def classify_coin(radius):
    if radius < RADIUS_THRESHOLD:
        return 1
    else:
        return 0.5

# ================================
# MAIN LOOP
# ================================
while True:
    success, image = cap.read()
    if not success:
        break

    imageCopy = image.copy()

    # --------------------------------
    # 1) YOUR preprocessing method
    # --------------------------------
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (5, 5), 0)

    _, binary = cv.threshold(blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    kernel = np.ones((5, 5), np.uint8)
    clean = cv.morphologyEx(binary, cv.MORPH_CLOSE, kernel, iterations=2)

    # --------------------------------
    # 2) HSV Mask
    # --------------------------------
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    hsvMask = cv.inRange(hsv, lowerHSV, upperHSV)
    hsvMask = cv.morphologyEx(hsvMask, cv.MORPH_CLOSE, kernel, iterations=2)

    # --------------------------------
    # 3) Combine (clean AND hsvMask)
    # --------------------------------
    combined = cv.bitwise_and(clean, hsvMask)

    # --------------------------------
    # 4) Hough Circle Detection
    # --------------------------------
    combinedBlur = cv.GaussianBlur(combined, (9, 9), 2)

    houghP2 = cv.getTrackbarPos("HoughP2", "Settings")
    minR = cv.getTrackbarPos("MinRadius", "Settings")
    maxR = cv.getTrackbarPos("MaxRadius", "Settings")

    circles = cv.HoughCircles(
        combinedBlur,
        cv.HOUGH_GRADIENT,
        dp=1.2,
        minDist=80,
        param1=120,
        param2=houghP2,
        minRadius=minR,
        maxRadius=maxR
    )

    # filter inner circles (important fix)
    filtered_circles = filter_inner_circles(circles, center_thresh=35)

    # --------------------------------
    # 5) Draw + Total Money
    # --------------------------------
    total_money = 0

    for x, y, r in filtered_circles:
        coin_value = classify_coin(r)
        total_money += coin_value

        cv.circle(imageCopy, (x, y), r, (0, 255, 0), 2)
        cv.circle(imageCopy, (x, y), 2, (0, 0, 255), 3)

        cv.putText(imageCopy, f"{coin_value} EGP", (x - 40, y),
                   cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    # --------------------------------
    # 6) Show Results
    # --------------------------------
    imgCount = np.zeros((480, 640, 3), np.uint8)
    cvzone.putTextRect(imgCount, f"EGP {total_money}", (100, 300),
                       scale=5, colorR=(0, 0, 255), thickness=5)

    imageStacked = cvzone.stackImages(
        [image, combined, imageCopy, imgCount], 2, 0.6
    )

    cvzone.putTextRect(imageStacked, f"Total: EGP {total_money}", (50, 50), colorR=(0, 0, 255))
    cv.imshow("EGP Coin Counter (HSV + Hough)", imageStacked)

    key = cv.waitKey(1)
    if key == ord("q"):
        break

cap.release()
cv.destroyAllWindows()
