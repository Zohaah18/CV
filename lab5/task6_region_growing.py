import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

os.makedirs("outputs", exist_ok=True)

img = cv2.imread("brain_mri.png", cv2.IMREAD_GRAYSCALE)


def region_growing(image, seed, threshold):
    h, w = image.shape
    visited = np.zeros((h, w), np.uint8)
    out = np.zeros((h, w), np.uint8)
    seed_val = int(image[seed[1], seed[0]])
    stack = [seed]
    visited[seed[1], seed[0]] = 1
    while stack:
        x, y = stack.pop()
        out[y, x] = 255
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < w and 0 <= ny < h and not visited[ny, nx]:
                visited[ny, nx] = 1
                if abs(int(image[ny, nx]) - seed_val) <= threshold:
                    stack.append((nx, ny))
    return out


seed_point = (130, 188)
mask = region_growing(img, seed_point, 25)
cv2.imwrite("outputs/task6_region_grown.png", mask)

overlay = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
overlay[mask == 255] = (0, 0, 255)

fig, ax = plt.subplots(1, 3, figsize=(15, 6.5))
ax[0].imshow(img, cmap="gray")
ax[0].plot(*seed_point, "g+", markersize=15, mew=2)
ax[0].set_title("Original + seed")
ax[1].imshow(mask, cmap="gray")
ax[1].set_title("Binary segmentation mask")
ax[2].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
ax[2].set_title("Overlay")
for a in ax:
    a.axis("off")
plt.tight_layout()
plt.savefig("outputs/task6_required_output.png", dpi=150, bbox_inches="tight")
plt.show()

thresholds = [15, 25, 35]
seeds = [(130, 188), (140, 130)]

fig, ax = plt.subplots(len(seeds), len(thresholds), figsize=(13, 10))
for i, sd in enumerate(seeds):
    for j, th in enumerate(thresholds):
        m = region_growing(img, sd, th)
        ax[i, j].imshow(m, cmap="gray")
        ax[i, j].set_title(f"seed={sd}, thresh={th} ({int(m.sum() / 255)} px)", fontsize=9)
        ax[i, j].axis("off")
plt.tight_layout()
plt.savefig("outputs/task6_parameter_study.png", dpi=150, bbox_inches="tight")
plt.show()

print("Why can changing the seed point significantly change the final segmented region?")
print("Region growing only absorbs neighbours that are both intensity-similar to the seed and spatially")
print("connected to it. Different seeds start from different reference intensities and lie in tissue")
print("compartments separated by anatomical boundaries, so the same threshold grows entirely different regions.")
