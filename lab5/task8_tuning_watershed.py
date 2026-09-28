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


def run_watershed(ratio):
    _, sure_fg = cv2.threshold(dist, ratio * dist.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    n, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    vis = color.copy()
    ws = cv2.watershed(vis, markers.copy())
    n_regions = len(np.unique(ws)) - 2
    vis[ws == -1] = [0, 0, 255]
    return n - 1, n_regions, vis


ratios = [0.15, 0.4, 0.5, 0.7]
results = [run_watershed(r) for r in ratios]

observations = [
    "Whole coin mass merges into one foreground blob (severe under-segmentation)",
    "Transitional: several touching clusters share markers, some coins still fused",
    "Every coin gets its own marker, touching coins cleanly separated",
    "Still fully separated, each coin keeps its own marker",
]

fig, ax = plt.subplots(1, 4, figsize=(22, 6))
for a, r, (n_fg, n_reg, vis) in zip(ax, ratios, results):
    a.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    a.set_title(f"dist thresh={r}\nmarkers={n_fg}, regions={n_reg}", fontsize=10)
    a.axis("off")
    cv2.imwrite(f"outputs/task8_watershed_ratio{r}.png", vis)
plt.tight_layout()
plt.savefig("outputs/task8_required_output.png", dpi=150, bbox_inches="tight")
plt.show()

print(f"{'Experiment':<12}{'Distance Threshold':<20}{'Foreground':<12}{'Regions':<10}Observed Result")
for i, (r, (n_fg, n_reg, _), obs) in enumerate(zip(ratios, results, observations), 1):
    print(f"{i:<12}{r:<20}{n_fg:<12}{n_reg:<10}{obs}")

print()
print("Best parameter range: distance-transform threshold of about 0.5 x dist.max() or higher.")
print("Below about 0.4, touching coins merge into shared markers.")
