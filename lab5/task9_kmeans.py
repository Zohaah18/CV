import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

os.makedirs("outputs", exist_ok=True)

img_bgr = cv2.imread("dog.jpeg")

Z = img_bgr.reshape((-1, 3)).astype(np.float32)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)

Ks = [2, 4, 6]
segmented = {}
for K in Ks:
    _, labels, centers = cv2.kmeans(Z, K, None, criteria, 5, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    seg = centers[labels.flatten()].reshape(img_bgr.shape)
    segmented[K] = seg
    cv2.imwrite(f"outputs/task9_kmeans_K{K}.png", seg)
    diff = np.mean(np.abs(img_bgr.astype(np.int16) - seg.astype(np.int16)))
    counts = np.bincount(labels.flatten(), minlength=K)
    print(f"K={K}: clusters={K}, mean abs difference from original={diff:.2f}")
    print(f"  cluster sizes (pixels): {counts.tolist()}")
    print(f"  cluster centres (BGR): {centers.tolist()}")

fig, ax = plt.subplots(1, 4, figsize=(20, 7.5))
ax[0].imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
ax[0].set_title("Original")
ax[0].axis("off")
for a, K in zip(ax[1:], Ks):
    a.imshow(cv2.cvtColor(segmented[K], cv2.COLOR_BGR2RGB))
    a.set_title(f"K={K}")
    a.axis("off")
plt.tight_layout()
plt.savefig("outputs/task9_required_output.png", dpi=150, bbox_inches="tight")
plt.show()

print()
print("K=2: collapses the scene into dark vs light only (heaviest colour simplification).")
print("K=4: recovers major regions (canopy, path/shadow, clothing, white coat).")
print("K=6: separates nearly every distinct material (smallest difference from original).")
