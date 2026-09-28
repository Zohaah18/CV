import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

os.makedirs("outputs", exist_ok=True)

color = cv2.imread("water_coins.jpg")
gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)

hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
otsu_thresh, otsu_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
cv2.imwrite("outputs/task3_otsu_mask.png", otsu_mask)
print(f"Otsu threshold: {otsu_thresh}")

fig, ax = plt.subplots(1, 3, figsize=(15, 5.5))
ax[0].imshow(gray, cmap="gray")
ax[0].set_title("Original")
ax[0].axis("off")
ax[1].plot(hist)
ax[1].axvline(otsu_thresh, color="r", linestyle="--")
ax[1].set_title(f"Histogram, Otsu T={otsu_thresh:.0f}")
ax[2].imshow(otsu_mask, cmap="gray")
ax[2].set_title("Otsu Binary Mask")
ax[2].axis("off")
plt.tight_layout()
plt.savefig("outputs/task3_required_output.png", dpi=150, bbox_inches="tight")
plt.show()

low_contrast = np.clip(128 + (gray.astype(np.float32) - 128) * 0.5, 0, 255).astype(np.uint8)
high_contrast = cv2.convertScaleAbs(gray, alpha=1.6, beta=-60)
blurred = cv2.GaussianBlur(gray, (7, 7), 0)

variants = {"Original": gray, "Low contrast": low_contrast, "High contrast": high_contrast, "Blurred": blurred}
for name, im in variants.items():
    t, _ = cv2.threshold(im, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    print(f"{name}: Otsu T = {t:.1f}")

print()
print("Why is Otsu useful when the programmer does not know the correct threshold beforehand?")
print("Otsu picks the threshold that maximises between-class variance from the image's own histogram,")
print("with no hand-tuned constant, so it adapts automatically to lighting and contrast changes.")
