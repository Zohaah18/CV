import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

os.makedirs("outputs", exist_ok=True)

color = cv2.imread("water_coins.jpg")
gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)

blurred = cv2.GaussianBlur(gray, (5, 5), 0)
_, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

kernel = np.ones((3, 3), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
sure_bg = cv2.dilate(opening, kernel, iterations=3)

dist = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
_, sure_fg = cv2.threshold(dist, 0.5 * dist.max(), 255, 0)
sure_fg = np.uint8(sure_fg)

unknown = cv2.subtract(sure_bg, sure_fg)

_, markers = cv2.connectedComponents(sure_fg)
markers = markers + 1
markers[unknown == 255] = 0

markers_ws = cv2.watershed(color.copy(), markers.copy())
n_regions = len(np.unique(markers_ws)) - 2

result = color.copy()
result[markers_ws == -1] = [0, 0, 255]
cv2.imwrite("outputs/task7_final_watershed.png", result)

stages = [gray, thresh, sure_bg, dist / dist.max() * 255, sure_fg, unknown, cv2.cvtColor(result, cv2.COLOR_BGR2RGB)]
titles = ["Original", "Threshold", "Sure Background", "Distance Transform", "Sure Foreground", "Unknown Region", "Final Watershed"]

fig, ax = plt.subplots(1, 7, figsize=(26, 5.4))
for a, im, t in zip(ax, stages, titles):
    a.imshow(im.astype(np.uint8) if im.ndim == 3 else im, cmap=None if im.ndim == 3 else "gray")
    a.set_title(t, fontsize=10)
    a.axis("off")
plt.tight_layout()
plt.savefig("outputs/task7_required_output.png", dpi=150, bbox_inches="tight")
plt.show()

print(f"Separated regions detected: {n_regions}")
