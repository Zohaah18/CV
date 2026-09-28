import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

os.makedirs("outputs", exist_ok=True)

img_bgr = cv2.imread("dog.jpeg")
gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

_, otsu_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

Z = img_bgr.reshape((-1, 3)).astype(np.float32)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
_, labels, centers = cv2.kmeans(Z, 5, None, criteria, 5, cv2.KMEANS_RANDOM_CENTERS)
kmeans_seg = np.uint8(centers)[labels.flatten()].reshape(img_bgr.shape)

edges = cv2.Canny(gray, 50, 150)

cv2.imwrite("outputs/task10_otsu.png", otsu_mask)
cv2.imwrite("outputs/task10_kmeans.png", kmeans_seg)
cv2.imwrite("outputs/task10_canny.png", edges)

fig, ax = plt.subplots(1, 4, figsize=(22, 7.8))
ax[0].imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
ax[0].set_title("Original")
ax[1].imshow(otsu_mask, cmap="gray")
ax[1].set_title("Method 1: Otsu")
ax[2].imshow(cv2.cvtColor(kmeans_seg, cv2.COLOR_BGR2RGB))
ax[2].set_title("Method 2: K-Means (K=5)")
ax[3].imshow(edges, cmap="gray")
ax[3].set_title("Method 3: Canny (50, 150)")
for a in ax:
    a.axis("off")
plt.tight_layout()
plt.savefig("outputs/task10_final_comparison.png", dpi=150, bbox_inches="tight")
plt.show()

print("Method 1 - Otsu")
print("  Assumption: bimodal intensity histogram (object vs background).")
print("  Identifies: bright sunlit areas vs dark shadow.")
print("  Incorrect: mixes the dog's white coat with sunlit leaves; no notion of objects.")
print("  Most influential parameter: none (threshold is automatic).")
print()
print("Method 2 - K-Means")
print("  Assumption: similar colours belong to the same object or material.")
print("  Identifies: dog's coat, clothing and path as fairly coherent regions.")
print("  Incorrect: dappled lighting splits one material into lit and shadow clusters.")
print("  Most influential parameter: K.")
print()
print("Method 3 - Canny")
print("  Assumption: object boundaries are strong intensity gradients.")
print("  Identifies: clean outline of the dog and the person.")
print("  Incorrect: dense clutter from the textured background foliage.")
print("  Most influential parameter: high threshold.")
print()
print("Conclusion: K-Means produced the most meaningful regions. The image is a multi-object,")
print("multi-material natural scene where colour similarity matches object identity far better")
print("than a single global intensity threshold.")
