import cv2
import numpy as np
import sys

# Constants
GOLD_AREA_THRESHOLD = 70
LOW_H = 12
HIGH_H = 19
LOW_S = 140
LOW_L = 100
HIGH_L = 190
GAMMA = 2.0


# Functions
def evaluate_gold_for_area(img_target, area_threshold):
    has_gold = False
    img_edged = cv2.Canny(img_target, 30, 200)
    contours, hierarchy = cv2.findContours(img_edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > area_threshold:
            has_gold = True
    return has_gold


def transform_img_hls(target_img, low_h, high_h, low_s, low_l, high_l, gamma):
    img = target_img.copy()
    img = (255 * (img / 255) ** gamma).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    color_rng = (np.array([low_h, low_l, low_s]), np.array([high_h, high_l, 255]))
    mask = cv2.inRange(img, *color_rng)
    img = cv2.bitwise_and(img, img, mask=mask)
    img = cv2.cvtColor(img, cv2.COLOR_HLS2BGR)
    img = cv2.erode(img, np.ones((1, 1)))
    img = cv2.dilate(img, np.ones((5, 5)))
    img = cv2.erode(img, np.ones((8, 8)))
    img = cv2.dilate(img, np.ones((8, 8)))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    return evaluate_gold_for_area(img, GOLD_AREA_THRESHOLD)


def execute_gold_detection(path):
    img_org = cv2.imread(path)
    img_org = cv2.resize(img_org, (640, 480))
    return transform_img_hls(img_org, LOW_H, HIGH_H, LOW_S, LOW_L, HIGH_L, GAMMA)


if __name__ == "__main__":
    execute_gold_detection(sys.argv[0])
