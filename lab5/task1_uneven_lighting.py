import os
import cv2
import matplotlib.pyplot as plt

os.makedirs("outputs", exist_ok=True)

img = cv2.imread("sudoku.png", cv2.IMREAD_GRAYSCALE)

T1, T2, T3 = 80, 120, 160
_, g1 = cv2.threshold(img, T1, 255, cv2.THRESH_BINARY)
_, g2 = cv2.threshold(img, T2, 255, cv2.THRESH_BINARY)
_, g3 = cv2.threshold(img, T3, 255, cv2.THRESH_BINARY)

adaptive = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 10)

cv2.imwrite("outputs/task1_global_T1.png", g1)
cv2.imwrite("outputs/task1_global_T2.png", g2)
cv2.imwrite("outputs/task1_global_T3.png", g3)
cv2.imwrite("outputs/task1_adaptive.png", adaptive)

titles = ["Original", f"Global T={T1}", f"Global T={T2}", f"Global T={T3}", "Adaptive"]
images = [img, g1, g2, g3, adaptive]

fig, ax = plt.subplots(1, 5, figsize=(22, 5))
for a, im, t in zip(ax, images, titles):
    a.imshow(im, cmap="gray")
    a.set_title(t)
    a.axis("off")
plt.tight_layout()
plt.savefig("outputs/task1_final_comparison.png", dpi=150, bbox_inches="tight")
plt.show()

print("Parameters: global T = 80, 120, 160; adaptive = Gaussian, blockSize=35, C=10")
print()
print("Why does a single threshold struggle when illumination changes across the image?")
print("A global threshold assumes one fixed cut point separates ink from paper everywhere.")
print("Uneven lighting shifts both ink and paper intensities in the shadowed region, so no single")
print("value is correct on both sides. Adaptive thresholding compares each pixel to its own local")
print("neighbourhood, so it stays correct across the shadow boundary.")
