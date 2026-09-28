import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

os.makedirs("outputs", exist_ok=True)

img_bgr = cv2.imread("smarties.png")
img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

mask_tight = cv2.inRange(img_hsv, np.array([0, 240, 240]), np.array([2, 255, 255]))

mask_final = cv2.bitwise_or(
    cv2.inRange(img_hsv, np.array([0, 120, 90]), np.array([10, 255, 255])),
    cv2.inRange(img_hsv, np.array([170, 120, 90]), np.array([180, 255, 255])),
)

extracted = cv2.bitwise_and(img_bgr, img_bgr, mask=mask_final)

cv2.imwrite("outputs/task4_mask_tight.png", mask_tight)
cv2.imwrite("outputs/task4_mask_final.png", mask_final)
cv2.imwrite("outputs/task4_extracted.png", extracted)

fig, ax = plt.subplots(1, 4, figsize=(20, 5.5))
ax[0].imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
ax[0].set_title("Original")
ax[1].imshow(mask_tight, cmap="gray")
ax[1].set_title(f"Too-restrictive mask ({mask_tight.sum() // 255} px)")
ax[2].imshow(mask_final, cmap="gray")
ax[2].set_title(f"Final mask ({mask_final.sum() // 255} px)")
ax[3].imshow(cv2.cvtColor(extracted, cv2.COLOR_BGR2RGB))
ax[3].set_title("Extracted")
for a in ax:
    a.axis("off")
plt.tight_layout()
plt.savefig("outputs/task4_required_output.png", dpi=150, bbox_inches="tight")
plt.show()

print("Range used (red): H 0-10 and 170-180, S 120-255, V 90-255")
print("Restrictive range: H 0-2, S 240-255, V 240-255")
print()
print("Why does the restrictive mask fail?")
print("It accepts only near-pure, fully saturated red at maximum brightness, but real candy surfaces have")
print("highlights and shaded edges outside that narrow band, so the mask is fragmented. Widening the")
print("saturation/value floor and covering red's hue wrap-around captures the whole object while still")
print("excluding the other colours.")
